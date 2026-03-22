import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        h = torch.matmul(adj, x)
        return self.linear(h)


class BiGCN(nn.Module):
    """
    Paper-style: Bottom-Up + Top-Down GCN
    """

    def __init__(self, in_dim=32, hidden_dim=64, num_classes=2):
        super().__init__()

        # Bottom-up
        self.gcn1 = GCNLayer(in_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)

        # Top-down
        self.gcn3 = GCNLayer(in_dim, hidden_dim)
        self.gcn4 = GCNLayer(hidden_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, adj, adj_rev):
        # Bottom-up
        h1 = F.relu(self.gcn1(x, adj))
        h1 = F.relu(self.gcn2(h1, adj))

        # Top-down
        h2 = F.relu(self.gcn3(x, adj_rev))
        h2 = F.relu(self.gcn4(h2, adj_rev))

        # Global pooling
        h1 = torch.mean(h1, dim=0)
        h2 = torch.mean(h2, dim=0)

        h = torch.cat([h1, h2], dim=0)

        return self.fc(h)