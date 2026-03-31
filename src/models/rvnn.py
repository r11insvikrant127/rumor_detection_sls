import torch
import torch.nn as nn


class RvNN(nn.Module):
    def __init__(self, vocab_size=5000, hidden_dim=100, num_classes=4):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Embedding matrix (paper-style sparse lookup)
        self.E = nn.Parameter(torch.randn(hidden_dim, vocab_size) * 0.1)

        # GRU parameters
        self.W_z = nn.Linear(hidden_dim, hidden_dim)
        self.U_z = nn.Linear(hidden_dim, hidden_dim)

        self.W_r = nn.Linear(hidden_dim, hidden_dim)
        self.U_r = nn.Linear(hidden_dim, hidden_dim)

        self.W_h = nn.Linear(hidden_dim, hidden_dim)
        self.U_h = nn.Linear(hidden_dim, hidden_dim)

        # Output layer
        self.out = nn.Linear(hidden_dim, num_classes)

    # -------------------------------------------------
    # NODE UPDATE (GRU)
    # -------------------------------------------------
    def node_forward(self, word, index, child_h):

        # Sparse embedding lookup
        xe = torch.matmul(self.E[:, index], word)  # [hidden_dim]

        h_tilde = torch.sum(child_h, dim=0)

        z = torch.sigmoid(self.W_z(xe) + self.U_z(h_tilde))
        r = torch.sigmoid(self.W_r(xe) + self.U_r(h_tilde))

        c = torch.tanh(self.W_h(xe) + self.U_h(h_tilde * r))
        h = z * h_tilde + (1 - z) * c

        return h

    # -------------------------------------------------
    # FORWARD (BOTTOM-UP)
    # -------------------------------------------------
    def forward(self, X_word, X_index, tree):

        device = self.E.device
        num_nodes = len(X_word)

        h = [None] * num_nodes

        # ---------------------------------
        # PRE-CONVERT (EFFICIENCY)
        # ---------------------------------
        X_word = [torch.tensor(w, dtype=torch.float32, device=device) for w in X_word]
        X_index = [torch.tensor(idx, dtype=torch.long, device=device) for idx in X_index]

        # ---------------------------------
        # LEAF NODES (PAPER-CORRECT ✅)
        # ---------------------------------
        num_leaves = len(X_word) - len(tree)

        for i in range(num_leaves):
            child_states = torch.zeros(1, self.hidden_dim, device=device)
            h[i] = self.node_forward(X_word[i], X_index[i], child_states)

        # ---------------------------------
        # INTERNAL NODES
        # ---------------------------------
        for row in tree:
            children = row[:-1]
            parent = row[-1]

            valid_children = [h[c] for c in children]


            if len(valid_children) == 0:
                raise ValueError(
                    f"Invalid tree: node {parent} has no children"
                )

            child_states = torch.stack(valid_children)

            h[parent] = self.node_forward(
                X_word[parent],
                X_index[parent],
                child_states
            )

        # ---------------------------------
        # ROOT
        # ---------------------------------
        root_state = h[len(X_word) - 1]

        return self.out(root_state).unsqueeze(0)