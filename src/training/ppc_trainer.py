import torch
import torch.nn as nn
import numpy as np

from src.preprocessing.tree_builder_ppc import TreeBuilderPPC


class PPCTrainer:

    def __init__(self, model, device="cuda", lr=1e-3, epochs=20, config=None):
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs

        self.criterion = None
        self.optimizer = torch.optim.Adadelta(self.model.parameters())

        self.tree_builder = TreeBuilderPPC()

        self.max_len = 40

        self.early_stop = True if config is None else config.get("early_stopping", True)
        self.patience = 5 if config is None else config.get("early_stopping_patience", 5)

        self.batch_size = 32

    # ==================================================
    # BUILD SEQUENCE
    # ==================================================
    def build_sequence(self, graph):

        nodes = sorted(
            graph.nodes(data=True),
            key=lambda x: x[1].get("time", 0)
        )

        features = []

        for _, attr in nodes:
            feat = attr.get("features", None)
            if feat is not None:
                features.append(feat)

        if len(features) == 0:
            return np.zeros((self.max_len, self.model.gru.input_size), dtype=np.float32)

        features = np.array(features, dtype=np.float32)

        # -------- TRUNCATE / PAD --------
        if len(features) >= self.max_len:
            features = features[:self.max_len]
        else:
            pad_len = self.max_len - len(features)
            indices = np.random.choice(len(features), pad_len, replace=True)
            extra = features[indices]
            features = np.concatenate([features, extra], axis=0)

        # -------- NORMALIZATION --------
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        features = (features - mean) / std

        return features

    # ==================================================
    # TRAIN
    # ==================================================
    def train(self, events_train, labels_train, events_val, labels_val):

        # -------- CLASS WEIGHTS --------
        class_counts = np.bincount(labels_train)
        weights = class_counts.sum() / (len(class_counts) * class_counts)

        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weights)

        print("PPC Class counts:", class_counts)
        print("PPC Class weights:", weights)

        best_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(self.epochs):

            # -------- SHUFFLE (IMPORTANT) --------
            perm = np.random.permutation(len(events_train))
            events_train = [events_train[i] for i in perm]
            labels_train = labels_train[perm]

            # -------- TRAIN --------
            self.model.train()
            total_loss = 0

            for i in range(0, len(events_train), self.batch_size):

                batch_graphs = events_train[i:i+self.batch_size]
                batch_labels = labels_train[i:i+self.batch_size]

                batch_features = [
                    self.build_sequence(graph) for graph in batch_graphs
                ]

                x = torch.tensor(batch_features, dtype=torch.float32).to(self.device)
                y = torch.tensor(batch_labels, dtype=torch.long).to(self.device)

                out = self.model(x)
                loss = self.criterion(out, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            num_batches = int(np.ceil(len(events_train) / self.batch_size))
            train_loss = total_loss / num_batches

            # -------- VALIDATION --------
            self.model.eval()
            val_loss = 0

            with torch.no_grad():
                for i in range(0, len(events_val), self.batch_size):

                    batch_graphs = events_val[i:i+self.batch_size]
                    batch_labels = labels_val[i:i+self.batch_size]

                    batch_features = [
                        self.build_sequence(graph) for graph in batch_graphs
                    ]

                    x = torch.tensor(batch_features, dtype=torch.float32).to(self.device)
                    y = torch.tensor(batch_labels, dtype=torch.long).to(self.device)

                    out = self.model(x)
                    loss = self.criterion(out, y)

                    val_loss += loss.item()

            num_val_batches = int(np.ceil(len(events_val) / self.batch_size))
            val_loss = val_loss / num_val_batches

            print(f"[PPC] Epoch {epoch+1} Train: {train_loss:.4f} | Val: {val_loss:.4f}")

            # -------- EARLY STOPPING --------
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = self.model.state_dict()
                no_improve = 0
            else:
                no_improve += 1

            if self.early_stop and no_improve >= self.patience:
                print(f"[PPC] Early stopping at epoch {epoch+1}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    # ==================================================
    # PREDICT
    # ==================================================
    def predict(self, events):

        self.model.eval()
        preds = []

        with torch.no_grad():
            for i in range(0, len(events), self.batch_size):

                batch_graphs = events[i:i+self.batch_size]

                batch_features = [
                    self.build_sequence(graph) for graph in batch_graphs
                ]

                x = torch.tensor(batch_features, dtype=torch.float32).to(self.device)

                out = self.model(x)
                preds.extend(torch.argmax(out, dim=1).cpu().numpy())

        return np.array(preds)