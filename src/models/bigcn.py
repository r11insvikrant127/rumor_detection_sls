import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean


class TDrumorGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim + in_dim, hidden_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x1 = x.float()
        x = self.conv1(x, edge_index)
        x2 = x  # pre-activation (paper-consistent)

        rootindex = data.rootindex
        batch = data.batch

        # ===== ROOT EXTEND (FIRST) =====
        root_extend = torch.zeros_like(x1, device=x.device)

        for b in range(batch.max().item() + 1):
            mask = (batch == b)
            root = rootindex[b].item()   # ✅ FIXED (NO ptr)
            root_extend[mask] = x1[root]

        x = torch.cat((x, root_extend), dim=1)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # ===== ROOT EXTEND (SECOND) =====
        root_extend = torch.zeros_like(x, device=x.device)

        for b in range(batch.max().item() + 1):
            mask = (batch == b)
            root = rootindex[b].item()   # ✅ FIXED
            root_extend[mask] = x2[root]

        x = torch.cat((x, root_extend), dim=1)

        x = scatter_mean(x, batch, dim=0)

        return x


class BUrumorGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim + in_dim, hidden_dim)

    def forward(self, data):
        x, edge_index = data.x, data.BU_edge_index

        x1 = x.float()
        x = self.conv1(x, edge_index)
        x2 = x  # pre-activation

        rootindex = data.rootindex
        batch = data.batch

        # ===== ROOT EXTEND (FIRST) =====
        root_extend = torch.zeros_like(x1, device=x.device)

        for b in range(batch.max().item() + 1):
            mask = (batch == b)
            root = rootindex[b].item()   # ✅ FIXED
            root_extend[mask] = x1[root]

        x = torch.cat((x, root_extend), dim=1)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # ===== ROOT EXTEND (SECOND) =====
        root_extend = torch.zeros_like(x, device=x.device)

        for b in range(batch.max().item() + 1):
            mask = (batch == b)
            root = rootindex[b].item()   # ✅ FIXED
            root_extend[mask] = x2[root]

        x = torch.cat((x, root_extend), dim=1)

        x = scatter_mean(x, batch, dim=0)

        return x


class BiGCN(nn.Module):
    def __init__(self, in_dim=5000, hidden_dim=64, num_classes=2):
        super().__init__()

        self.TD = TDrumorGCN(in_dim, hidden_dim)
        self.BU = BUrumorGCN(in_dim, hidden_dim)

        # (TD + BU) → each gives 2*hidden_dim
        self.fc = nn.Linear(hidden_dim * 4, num_classes)

    def forward(self, data):
        td = self.TD(data)
        bu = self.BU(data)

        x = torch.cat((bu, td), dim=1)
        x = self.fc(x)

        return x