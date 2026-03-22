import torch
import torch.nn as nn
import numpy as np
from src.preprocessing.tree_builder import TreeBuilder
from src.preprocessing.graph_builder import build_node_features


class RvNNTrainer:

    def __init__(self, model, device="cuda", lr=1e-3, epochs=20, config=None):
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.tree_builder = TreeBuilder()

        # 🔥 Early stopping config
        self.early_stop = True if config is None else config.get("early_stopping", True)
        self.patience = 5 if config is None else config.get("early_stopping_patience", 5)

    def train(self, events_train, labels_train, events_val, labels_val):

        best_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(self.epochs):

            # -------- TRAIN --------
            self.model.train()
            total_loss = 0

            for event, label in zip(events_train, labels_train):

                graph = event

                nodes = list(graph.nodes())
                features = build_node_features(graph, nodes)

                x = torch.tensor(features, dtype=torch.float32).to(self.device)
                label = torch.tensor([label]).to(self.device)

                out = self.model(x)
                loss = self.criterion(out, label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            train_loss = total_loss / len(events_train)

            # -------- VALIDATION --------
            self.model.eval()
            val_loss = 0

            with torch.no_grad():
                for event, label in zip(events_val, labels_val):

                    graph = event

                    nodes = list(graph.nodes())
                    features = build_node_features(graph, nodes)

                    x = torch.tensor(features, dtype=torch.float32).to(self.device)
                    label = torch.tensor([label]).to(self.device)

                    out = self.model(x)
                    loss = self.criterion(out, label)

                    val_loss += loss.item()

            val_loss = val_loss / len(events_val)

            print(f"[RvNN] Epoch {epoch+1} Train: {train_loss:.4f} | Val: {val_loss:.4f}")

            # -------- EARLY STOPPING --------
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = self.model.state_dict()
                no_improve = 0
            else:
                no_improve += 1

            if self.early_stop and no_improve >= self.patience:
                print(f"[RvNN] Early stopping at epoch {epoch+1}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def predict(self, events):

        self.model.eval()
        preds = []

        with torch.no_grad():
            for event in events:

                graph = event

                nodes = list(graph.nodes())
                features = build_node_features(graph, nodes)

                x = torch.tensor(features, dtype=torch.float32).to(self.device)

                out = self.model(x)
                preds.append(torch.argmax(out).item())

        return np.array(preds)