#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from typing import Any

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model.data_parallel import get_data_parallel
from model.mnist_convnet import MNISTConvNet
from model.mnist_mlp import MNISTMlp
from running_log import RunningLog


def load_epoch(save_path: str, epoch: int) -> Any:
    print('loading from epoch.%04d.pth' % epoch)
    return torch.load(os.path.join(save_path, 'epoch.%04d.pth' % epoch),
                      map_location='cpu')


def load_dataset(train: bool = True) -> datasets.MNIST:
    return datasets.MNIST('data', train=train, download=True,
                          transform=transforms.ToTensor())


def eval_model(model: nn.Module, data_loader: DataLoader,
               device: torch.device) -> float:
    total_count, correct_count = 0, 0
    for data in tqdm(data_loader, desc='Eval'):
        data = [x.to(device) for x in data]
        total_count += data[0].size(0)
        output = model(data[0])
        correct_count += (torch.argmax(output, dim=1) == data[1]).sum().item()
    return correct_count / total_count


def main():
    parser = argparse.ArgumentParser()
    # Common Options
    parser.add_argument('--model', choices=['MLP', 'ConvNet'], default='MLP',
                        help='model to run')
    parser.add_argument('--task', choices=['train', 'test'], default='train',
                        help='task to run')
    parser.add_argument('--save_path', help='path for saving models and codes',
                        default='save/test')
    parser.add_argument('--gpu', type=lambda x: list(map(int, x.split(','))),
                        default=[], help="GPU ids separated by `,'")
    parser.add_argument('--load', type=int, default=0,
                        help='load module training at give epoch')
    parser.add_argument('--epoch', type=int, default=200, help='epoch to train')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.003,
                        help='learning rate')
    parser.add_argument('--log_every_iter', type=int, default=100,
                        help='log loss every numbers of iteration')
    parser.add_argument('--test_every_epoch', type=int, default=5,
                        help='run test every numbers of epoch; '
                             '0 for disabling')
    parser.add_argument('--save_every_epoch', type=int, default=10,
                        help='save model every numbers of epoch; '
                             '0 for disabling')
    parser.add_argument('--comment', default='', help='comment for tensorboard')
    # Model options
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='hidden layer size for MLP')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout ratio')
    parser.add_argument('--activation', choices=['relu', 'sigmoid'],
                        default='relu', help='activation function to use')
    parser.add_argument('--shallower', action='store_true')
    # Build model
    args = parser.parse_args()
    running_log = RunningLog(args.save_path)
    running_log.set('parameters', vars(args))
    os.makedirs(args.save_path, exist_ok=True)
    if args.model == 'MLP':
        model = MNISTMlp(hidden_size=args.hidden_size, dropout=args.dropout,
                         shallower=args.shallower)
    else:
        model = MNISTConvNet(hidden_size=args.hidden_size, dropout=args.dropout,
                             activation=args.activation)
    model: nn.Module = get_data_parallel(model, args.gpu)
    device = torch.device("cuda:%d" % args.gpu[0] if args.gpu else "cpu")
    optimizer_state_dict = None
    if args.load > 0:
        model_state_dict, optimizer_state_dict = \
            load_epoch(args.save_path, args.load)
        model.load_state_dict(model_state_dict)
    model.to(device)
    running_log.set('state', 'interrupted')
    # Start training or testing
    if args.task == 'train':
        model.train()
        train_data_loader = DataLoader(load_dataset(train=True),
                                       batch_size=args.batch_size,
                                       shuffle=True)
        test_data_loader = None
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
        if optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)
        writer = SummaryWriter(comment=args.comment or
                                       os.path.basename(args.save_path))
        step = 0
        for epoch in tqdm(range(args.load + 1, args.epoch + 1), desc='Epoch'):
            losses = []
            for iter, data in enumerate(tqdm(train_data_loader, desc='Iter'),
                                        1):
                data = [x.to(device) for x in data]
                output = model(data[0])
                loss = F.nll_loss(output, data[1])
                losses.append(loss.item())
                writer.add_scalar('train/loss', loss.item(), step)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if iter % args.log_every_iter == 0:
                    # noinspection PyStringFormat
                    tqdm.write('epoch:[%d/%d] iter:[%d/%d] Loss=%.5f' %
                               (epoch, args.epoch, iter, len(train_data_loader),
                                np.mean(losses)))
                    losses = []
                step += 1
            if args.test_every_epoch and epoch % args.test_every_epoch == 0:
                if test_data_loader is None:
                    test_data_loader = DataLoader(load_dataset(train=False),
                                                  batch_size=args.batch_size,
                                                  shuffle=False)
                model.eval()
                acc = eval_model(model, test_data_loader, device)
                tqdm.write('Accuracy=%f' % acc)
                writer.add_scalar('eval/acc', acc, epoch)
                model.train()
            if args.save_every_epoch and epoch % args.save_every_epoch == 0:
                tqdm.write('saving to epoch.%04d.pth' % epoch)
                torch.save((model.state_dict(), optimizer.state_dict()),
                           os.path.join(args.save_path,
                                        'epoch.%04d.pth' % epoch))
    elif args.task == 'test':
        model.eval()
        test_data_loader = DataLoader(load_dataset(train=False),
                                      batch_size=args.batch_size,
                                      shuffle=False)
        acc = eval_model(model, test_data_loader, device)
        print('Accuracy=%f' % acc)
    running_log.set('state', 'succeeded')


if __name__ == '__main__':
    main()

