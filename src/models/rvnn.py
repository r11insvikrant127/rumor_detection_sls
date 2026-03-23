import torch
import torch.nn as nn
import networkx as nx


# =====================================================
# TREE GRU CELL (Top-Down)
# =====================================================
class TreeGRUCell(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.W_r = nn.Linear(input_dim, hidden_dim)
        self.U_r = nn.Linear(hidden_dim, hidden_dim)

        self.W_z = nn.Linear(input_dim, hidden_dim)
        self.U_z = nn.Linear(hidden_dim, hidden_dim)

        self.W_h = nn.Linear(input_dim, hidden_dim)
        self.U_h = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, h_parent):

        r = torch.sigmoid(self.W_r(x) + self.U_r(h_parent))
        z = torch.sigmoid(self.W_z(x) + self.U_z(h_parent))

        h_tilde = torch.tanh(self.W_h(x) + self.U_h(h_parent * r))
        h = (1 - z) * h_parent + z * h_tilde

        return h


# =====================================================
# TOP-DOWN RvNN (PAPER-FAITHFUL)
# =====================================================
class RvNN(nn.Module):

    def __init__(self, input_dim=2, hidden_dim=64):
        super().__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.tree_gru = TreeGRUCell(hidden_dim, hidden_dim)

        # output only uses pooled leaf representation
        self.out = nn.Linear(hidden_dim, 2)

    def forward(self, graph, features):

        # embed features
        x = self.embedding(features)

        # store hidden states
        h = {}

        # top-down traversal (root → leaves)
        nodes = list(nx.topological_sort(graph))

        for node in nodes:

            parents = list(graph.predecessors(node))

            # root node
            if len(parents) == 0:
                h_parent = torch.zeros_like(x[node])
            else:
                h_parent = h[parents[0]]

            h[node] = self.tree_gru(x[node], h_parent)

        # get leaf nodes
        leaves = [n for n in graph.nodes() if graph.out_degree(n) == 0]

        # stack leaf representations
        h_leaves = torch.stack([h[n] for n in leaves])

        # max pooling (CRITICAL — from paper)
        h_pool, _ = torch.max(h_leaves, dim=0)

        # classification
        out = self.out(h_pool)

        return out.unsqueeze(0)