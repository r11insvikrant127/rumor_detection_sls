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
    Bi-Directional GCN for Rumor Detection

    - Top-Down (Propagation)
    - Bottom-Up (Dispersion)
    - Root Feature Enhancement
    """

    def __init__(self, in_dim=32, hidden_dim=64, num_classes=2):
        super().__init__()

        # Top-Down
        self.td_gcn1 = GCNLayer(in_dim, hidden_dim)
        self.td_gcn2 = GCNLayer(hidden_dim + in_dim, hidden_dim)

        # Bottom-Up
        self.bu_gcn1 = GCNLayer(in_dim, hidden_dim)
        self.bu_gcn2 = GCNLayer(hidden_dim + in_dim, hidden_dim)

        # Classifier
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, adj, adj_rev):
        """
        x: [N, in_dim]
        adj: parent -> child
        adj_rev: child -> parent
        """

        # ---------------- ROOT FEATURE ----------------
        # assumes root node is first
        root = x[0].unsqueeze(0)
        root_expand = root.repeat(x.size(0), 1)

        # ---------------- TOP-DOWN ----------------
        td = F.relu(self.td_gcn1(x, adj))
        td = torch.cat([td, root_expand], dim=1)
        td = F.relu(self.td_gcn2(td, adj))

        # ---------------- BOTTOM-UP ----------------
        bu = F.relu(self.bu_gcn1(x, adj_rev))
        bu = torch.cat([bu, root_expand], dim=1)
        bu = F.relu(self.bu_gcn2(bu, adj_rev))

        # ---------------- POOLING ----------------
        td = torch.mean(td, dim=0)
        bu = torch.mean(bu, dim=0)

        # ---------------- FUSION ----------------
        h = torch.cat([td, bu], dim=0)

        return self.fc(h)