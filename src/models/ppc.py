import torch
import torch.nn as nn
import torch.nn.functional as F


class PPC(nn.Module):
    def __init__(self, input_dim=2):
        super().__init__()

        self.conv = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
        self.rnn = nn.GRU(32, 64, batch_first=True)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.conv(x))
        x = x.transpose(1, 2)

        _, h = self.rnn(x)
        return self.fc(h.squeeze(0))