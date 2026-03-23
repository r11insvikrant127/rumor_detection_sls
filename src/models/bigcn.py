import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================
# BASIC GCN LAYER
# =====================================================
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        h = torch.matmul(adj, x)
        return self.linear(h)


# =====================================================
# BI-GCN MODEL
# =====================================================
class BiGCN(nn.Module):
    """
    Bi-Directional GCN for Rumor Detection

    ✔ Top-Down (Propagation)
    ✔ Bottom-Up (Dispersion)
    ✔ Root Feature Enhancement
    ✔ Input Projection (for TF-IDF stability)
    ✔ Dropout (paper uses this)
    """

    def __init__(self, in_dim=3000, hidden_dim=64, num_classes=2):
        super().__init__()

        # ✅ INPUT PROJECTION (VERY IMPORTANT)
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # ---------------- TOP-DOWN ----------------
        self.td_gcn1 = GCNLayer(hidden_dim, hidden_dim)
        self.td_gcn2 = GCNLayer(hidden_dim + hidden_dim, hidden_dim)

        # ---------------- BOTTOM-UP ----------------
        self.bu_gcn1 = GCNLayer(hidden_dim, hidden_dim)
        self.bu_gcn2 = GCNLayer(hidden_dim + hidden_dim, hidden_dim)

        # ---------------- REGULARIZATION ----------------
        self.dropout = nn.Dropout(0.5)

        # ---------------- CLASSIFIER ----------------
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, adj, adj_rev):
        """
        x: [N, in_dim]
        adj: parent -> child
        adj_rev: child -> parent
        """

        # =====================================================
        # INPUT PROJECTION
        # =====================================================
        x = self.input_proj(x)   # [N, hidden_dim]

        # =====================================================
        # ROOT FEATURE (root is first node)
        # =====================================================
        root = x[0].unsqueeze(0)
        root_expand = root.repeat(x.size(0), 1)

        # =====================================================
        # TOP-DOWN GCN
        # =====================================================
        td = F.relu(self.td_gcn1(x, adj))
        td = self.dropout(td)

        td = torch.cat([td, root_expand], dim=1)
        td = F.relu(self.td_gcn2(td, adj))
        td = self.dropout(td)

        # =====================================================
        # BOTTOM-UP GCN
        # =====================================================
        bu = F.relu(self.bu_gcn1(x, adj_rev))
        bu = self.dropout(bu)

        bu = torch.cat([bu, root_expand], dim=1)
        bu = F.relu(self.bu_gcn2(bu, adj_rev))
        bu = self.dropout(bu)

        # =====================================================
        # POOLING (graph-level representation)
        # =====================================================
        td = torch.mean(td, dim=0)
        bu = torch.mean(bu, dim=0)

        # =====================================================
        # FUSION
        # =====================================================
        h = torch.cat([td, bu], dim=0)

        return self.fc(h)