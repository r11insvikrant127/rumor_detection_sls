import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from src.preprocessing.graph_builder_bigcn import (
    build_raw_adjacency,
    normalize_adjacency,
    build_node_features
)
from src.training.evaluator import Evaluator


# =====================================================
# DROPEDGE (PAPER-CORRECT)
# =====================================================
def drop_edge(adj, drop_rate=0.2):
    adj = adj.copy()
    edge_indices = np.argwhere(adj > 0)

    num_edges = len(edge_indices)
    num_drop = int(num_edges * drop_rate)

    if num_drop == 0:
        return adj

    drop_idx = np.random.choice(num_edges, num_drop, replace=False)

    for i in drop_idx:
        u, v = edge_indices[i]
        if u != v:  # keep self-loops
            adj[u, v] = 0

    return adj


# =====================================================
# TRAINER
# =====================================================
class BiGCNTrainer:

    def __init__(self, model, device="cuda", lr=1e-3, epochs=20, config=None):
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs

        # ✅ L2 Regularization (paper)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=1e-4
        )

        self.early_stop = True if config is None else config.get("early_stopping", True)
        self.patience = 5 if config is None else config.get("early_stopping_patience", 5)

    # =====================================================
    # HELPER: ROOT-FIRST NODE ORDER
    # =====================================================
    def _get_ordered_nodes(self, graph):
        nodes = list(graph.nodes())

        root_candidates = [n for n in nodes if graph.nodes[n]["depth"] == 0]

        if len(root_candidates) == 0:
            root_node = nodes[0]  # fallback
        else:
            root_node = root_candidates[0]

        nodes.remove(root_node)
        return [root_node] + nodes

    # =====================================================
    # TRAIN
    # =====================================================
    def train(self, events_train, labels_train, events_val, labels_val):

        best_loss = float("inf")
        best_state = None
        no_improve = 0

        # ✅ CLASS WEIGHTS
        class_counts = np.bincount(labels_train)
        weights = 1.0 / (class_counts + 1e-6)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

        self.criterion = nn.CrossEntropyLoss(weight=weights)

        for epoch in range(self.epochs):

            # ---------------- TRAIN ----------------
            self.model.train()
            total_loss = 0

            for event, label in zip(events_train, labels_train):

                graph = event
                nodes = self._get_ordered_nodes(graph)

                # ---- adjacency (paper correct) ----
                adj_raw = build_raw_adjacency(graph, nodes)
                adj_dropped = drop_edge(adj_raw, 0.2)

                adj = normalize_adjacency(adj_dropped)
                adj_rev = adj.T

                # ---- features ----
                features = build_node_features(graph, nodes)

                x = torch.tensor(features, dtype=torch.float32).to(self.device)
                adj = torch.tensor(adj, dtype=torch.float32).to(self.device)
                adj_rev = torch.tensor(adj_rev, dtype=torch.float32).to(self.device)

                label = torch.tensor([label], dtype=torch.long).to(self.device)

                out = self.model(x, adj, adj_rev).unsqueeze(0)

                loss = self.criterion(out, label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            train_loss = total_loss / len(events_train)

            # ---------------- VALIDATION ----------------
            self.model.eval()
            val_loss = 0

            with torch.no_grad():
                for event, label in zip(events_val, labels_val):

                    graph = event
                    nodes = self._get_ordered_nodes(graph)

                    # ---- IMPORTANT: NO DropEdge in validation ----
                    adj_raw = build_raw_adjacency(graph, nodes)
                    adj = normalize_adjacency(adj_raw)
                    adj_rev = adj.T

                    features = build_node_features(graph, nodes)

                    x = torch.tensor(features, dtype=torch.float32).to(self.device)
                    adj = torch.tensor(adj, dtype=torch.float32).to(self.device)
                    adj_rev = torch.tensor(adj_rev, dtype=torch.float32).to(self.device)

                    label = torch.tensor([label], dtype=torch.long).to(self.device)

                    out = self.model(x, adj, adj_rev).unsqueeze(0)
                    loss = self.criterion(out, label)

                    val_loss += loss.item()

            val_loss = val_loss / len(events_val)

            print(f"[BiGCN] Epoch {epoch+1} Train: {train_loss:.4f} | Val: {val_loss:.4f}")

            # ---------------- EARLY STOPPING ----------------
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = self.model.state_dict()
                no_improve = 0
            else:
                no_improve += 1

            if self.early_stop and no_improve >= self.patience:
                print(f"[BiGCN] Early stopping at epoch {epoch+1}")
                break

        # restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

    # =====================================================
    # PREDICT
    # =====================================================
    def predict(self, events):

        self.model.eval()
        preds = []

        with torch.no_grad():
            for event in events:

                graph = event
                nodes = self._get_ordered_nodes(graph)

                adj_raw = build_raw_adjacency(graph, nodes)
                adj = normalize_adjacency(adj_raw)
                adj_rev = adj.T

                features = build_node_features(graph, nodes)

                x = torch.tensor(features, dtype=torch.float32).to(self.device)
                adj = torch.tensor(adj, dtype=torch.float32).to(self.device)
                adj_rev = torch.tensor(adj_rev, dtype=torch.float32).to(self.device)

                out = self.model(x, adj, adj_rev)
                preds.append(torch.argmax(out).item())

        return np.array(preds)