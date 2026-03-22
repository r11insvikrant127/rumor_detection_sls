import torch
import torch.nn as nn


class RvNN(nn.Module):
    """
    Bottom-up tree model
    """

    def __init__(self, input_dim=2, hidden_dim=64):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.fc(x)
        x = x.unsqueeze(0)  # batch=1

        _, h = self.gru(x)
        return self.out(h.squeeze(0))