import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean


def drop_edge(edge_index, drop_rate=0.2):
    if edge_index.size(1) == 0:
        return edge_index
    mask = torch.rand(edge_index.size(1), device=edge_index.device) > drop_rate
    return edge_index[:, mask]


class TDrumorGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim + in_dim, hidden_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        if self.training:
            edge_index = drop_edge(edge_index)

        batch = data.batch
        rootindex = data.rootindex
        ptr = data.ptr

        x0 = x

        # ===== Layer 1 =====
        H1 = self.conv1(x0, edge_index)
        H1 = F.relu(H1)

        root_extend = torch.zeros_like(x0)
        ptr = data.ptr

        for b in range(batch.max().item() + 1):
            start = ptr[b]
            root = rootindex[b].item() + start
            # ✅ SAFE GUARD
            if root >= x0.size(0):
                continue
            mask = (batch == b)
            root_extend[mask] = x0[root]

        H1 = torch.cat([H1, root_extend], dim=1)
        H1 = F.dropout(H1, p=0.5, training=self.training)

        # ===== Layer 2 =====
        H2 = self.conv2(H1, edge_index)
        H2 = F.relu(H2)

        root_extend = torch.zeros_like(H2)
        ptr = data.ptr

        for b in range(batch.max().item() + 1):
            start = ptr[b]
            root = rootindex[b].item() + start
            # ✅ SAFE GUARD
            if root >= x0.size(0):
                continue
            mask = (batch == b)
            root_extend[mask] = H1[root]

        H2 = torch.cat([H2, root_extend], dim=1)

        out = scatter_mean(H2, batch, dim=0)
        return out


class BUrumorGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim + in_dim, hidden_dim)

    def forward(self, data):
        x, edge_index = data.x, data.BU_edge_index

        if self.training:
            edge_index = drop_edge(edge_index)

        batch = data.batch
        rootindex = data.rootindex
        ptr = data.ptr

        x0 = x

        # ===== Layer 1 =====
        H1 = self.conv1(x0, edge_index)
        H1 = F.relu(H1)

        root_extend = torch.zeros_like(x0)
        ptr = data.ptr

        for b in range(batch.max().item() + 1):
            start = ptr[b]
            root = rootindex[b].item() + start
            # ✅ SAFE GUARD
            if root >= x0.size(0):
                continue
            mask = (batch == b)
            root_extend[mask] = x0[root]

        H1 = torch.cat([H1, root_extend], dim=1)
        H1 = F.dropout(H1, p=0.5, training=self.training)

        # ===== Layer 2 =====
        H2 = self.conv2(H1, edge_index)
        H2 = F.relu(H2)

        root_extend = torch.zeros_like(H2)
        ptr = data.ptr

        for b in range(batch.max().item() + 1):
            start = ptr[b]
            root = rootindex[b].item() + start
            # ✅ SAFE GUARD
            if root >= x0.size(0):
                continue
            mask = (batch == b)
            root_extend[mask] = H1[root]

        H2 = torch.cat([H2, root_extend], dim=1)

        out = scatter_mean(H2, batch, dim=0)
        return out


class BiGCN(nn.Module):
    def __init__(self, in_dim=5000, hidden_dim=64, num_classes=2):
        super().__init__()
        self.TD = TDrumorGCN(in_dim, hidden_dim)
        self.BU = BUrumorGCN(in_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim * 4, num_classes)

    def forward(self, data):
        td = self.TD(data)
        bu = self.BU(data)

        x = torch.cat((td, bu), dim=1)
        x = self.fc(x)

        return x