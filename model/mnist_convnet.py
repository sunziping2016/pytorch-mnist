import torch.nn as nn
import torch.nn.functional as F

from model import choices


class MNISTConvNet(nn.Module):
    def __init__(self, hidden_size: int = 512, dropout: float = 0.2,
                 activation: str = 'relu'):
        super(MNISTConvNet, self).__init__()
        self.activation = choices.activations_choices[activation]
        self.conv1 = nn.Conv2d(1, 20, 5, 1) # output (n, 20, 24, 24)
        self.dropout1 = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(20, 50, 5, 1) # output (n, 50, 8, 8)
        self.dropout2 = nn.Dropout2d(dropout)
        self.fc1 = nn.Linear(50 * 4 * 4, hidden_size)
        self.dropout3 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.dropout1(F.max_pool2d(x, 2, 2))
        x = self.activation(self.conv2(x))
        x = self.dropout2(F.max_pool2d(x, 2, 2))
        x = x.view(-1, 50 * 4 * 4)
        x = self.dropout3(self.activation(self.fc1(x)))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x
