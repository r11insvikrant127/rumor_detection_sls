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
# TOP-DOWN RvNN (FIXED + STABLE)
# =====================================================
class RvNN(nn.Module):

    def __init__(self, input_dim=3000, hidden_dim=100, num_classes=2):
        super().__init__()

        self.hidden_dim = hidden_dim

        # TF-IDF → embedding
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # Tree GRU
        self.tree_gru = TreeGRUCell(hidden_dim, hidden_dim)

        # Learnable root initial state (better than zeros)
        self.root_state = nn.Parameter(torch.zeros(hidden_dim))

        # Output layer
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, graph, features):
        """
        graph: networkx DiGraph (tree)
        features: tensor [num_nodes, input_dim]
        """

        # -------------------------------------------------
        # 1. Embed input features
        # -------------------------------------------------
        x = self.embedding(features)   # [N, hidden_dim]

        # -------------------------------------------------
        # 2. Topological order (root → leaves)
        # -------------------------------------------------
        nodes = list(nx.topological_sort(graph))

        # CRITICAL: map node → index
        node_to_idx = {node: i for i, node in enumerate(nodes)}

        # store hidden states
        h = {}

        # -------------------------------------------------
        # 3. Top-down recursion
        # -------------------------------------------------
        for node in nodes:

            idx = node_to_idx[node]
            parents = list(graph.predecessors(node))

            # Root node
            if len(parents) == 0:
                h_parent = self.root_state
            else:
                parent = parents[0]   # tree assumption
                h_parent = h[parent]

            h[node] = self.tree_gru(x[idx], h_parent)

        # -------------------------------------------------
        # 4. Get leaf nodes
        # -------------------------------------------------
        leaves = [n for n in graph.nodes() if graph.out_degree(n) == 0]

        # -------------------------------------------------
        # 5. Max pooling over leaf representations
        # -------------------------------------------------
        h_leaves = torch.stack([h[n] for n in leaves])  # [num_leaves, hidden_dim]
        h_pool, _ = torch.max(h_leaves, dim=0)

        # -------------------------------------------------
        # 6. Classification
        # -------------------------------------------------
        out = self.out(h_pool)   # [num_classes]

        return out.unsqueeze(0)  # [1, num_classes]
