import torch.nn as nn
import torch.nn.functional as F


class MNISTMlp(nn.Module):
    def __init__(self, hidden_size: int = 512, dropout: float = 0.2,
                 shallower: bool = False):
        super(MNISTMlp, self).__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        if not shallower:
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.dropout1(F.relu(self.fc1(x)))
        if hasattr(self, 'fc2'):
            x = self.dropout2(F.relu(self.fc2(x)))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x
