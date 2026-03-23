import torch
import torch.nn as nn
import numpy as np

from src.preprocessing.tree_builder_ppc import TreeBuilderPPC


class PPCTrainer:

    def __init__(self, model, device="cuda", lr=1e-3, epochs=20, config=None):
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adadelta(self.model.parameters())

        self.tree_builder = TreeBuilderPPC()

        # -------- PPC CONFIG --------
        self.max_len = 40  # paper: ~30–40

        # -------- Early stopping --------
        self.early_stop = True if config is None else config.get("early_stopping", True)
        self.patience = 5 if config is None else config.get("early_stopping_patience", 5)

    # ==================================================
    # 🔥 BUILD SEQUENCE (MOST IMPORTANT PART)
    # ==================================================
    def build_sequence(self, graph):
        """
        Convert graph → time-ordered fixed-length sequence
        """

        # sort by propagation time
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
            # fallback (rare)
            return np.zeros((self.max_len, self.model.gru.input_size), dtype=np.float32)

        features = np.array(features, dtype=np.float32)

        # -------- TRUNCATE --------
        if len(features) >= self.max_len:
            features = features[:self.max_len]
        else:
            pad_len = self.max_len - len(features)
            indices = np.random.choice(len(features), pad_len, replace=True)
            extra = features[indices]
            features = np.concatenate([features, extra], axis=0)

        # -------- NORMALIZATION (AFTER FINAL SEQUENCE) --------
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        features = (features - mean) / std

        return features

            

    # ==================================================
    # TRAIN
    # ==================================================
    def train(self, events_train, labels_train, events_val, labels_val):

        best_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(self.epochs):

            # -------- TRAIN --------
            self.model.train()
            total_loss = 0

            for graph, label in zip(events_train, labels_train):

                features = self.build_sequence(graph)

                x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
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
                for graph, label in zip(events_val, labels_val):

                    features = self.build_sequence(graph)

                    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
                    label = torch.tensor([label]).to(self.device)

                    out = self.model(x)
                    loss = self.criterion(out, label)

                    val_loss += loss.item()

            val_loss = val_loss / len(events_val)

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
            for graph in events:

                features = self.build_sequence(graph)

                x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

                out = self.model(x)
                preds.append(torch.argmax(out).item())

        return np.array(preds)