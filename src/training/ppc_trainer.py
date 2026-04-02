import torch
import torch.nn as nn
import numpy as np

from src.preprocessing.tree_builder_ppc import PPCPreprocessor


class PPCTrainer:

    def __init__(self, model, device="cuda", lr=1e-3, epochs=200, config=None):

        self.model = model.to(device)
        self.device = device
        self.epochs = epochs

        self.optimizer = torch.optim.Adadelta(self.model.parameters())

        self.preprocessor = PPCPreprocessor(max_len=40)

        self.batch_size = 32

        self.early_stop = True if config is None else config.get("early_stopping", True)
        self.patience = 10 if config is None else config.get("early_stopping_patience", 10)

        self.criterion = None

    # ==================================================
    # BUILD SEQUENCE (DIRECT FROM PHEME JSON)
    # ==================================================
    def build_sequence(self, data):
        seq = self.preprocessor.process_thread(data)

        if seq is None:
            return np.zeros((40, 8), dtype=np.float32)

        # -------- NORMALIZATION (OPTIONAL BUT GOOD) --------
        seq[:, 2] = np.log1p(seq[:, 2])   # followers
        seq[:, 3] = np.log1p(seq[:, 3])   # friends
        seq[:, 4] = np.log1p(seq[:, 4])   # statuses

        seq[:, 5] = seq[:, 5] / 3650.0    # reg age (~10 yrs)
        seq[:, 0] = seq[:, 0] / 100.0     # description length
        seq[:, 1] = seq[:, 1] / 50.0      # username length

        return seq

    # ==================================================
    # TRAIN
    # ==================================================
    def train(self, events_train, labels_train, events_val, labels_val):

        # -------- CLASS WEIGHTS --------
        class_counts = np.bincount(labels_train, minlength=2)
        weights = class_counts.sum() / (len(class_counts) * class_counts)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

        self.criterion = nn.CrossEntropyLoss(weight=weights)

        print("PPC Class counts:", class_counts)
        print("PPC Class weights:", weights)

        best_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(self.epochs):

            # -------- SHUFFLE --------
            perm = np.random.permutation(len(events_train))
            events_train = [events_train[i] for i in perm]
            labels_train = labels_train[perm]

            self.model.train()
            total_loss = 0

            # -------- TRAIN LOOP --------
            for i in range(0, len(events_train), self.batch_size):

                batch_data = events_train[i:i+self.batch_size]
                batch_labels = labels_train[i:i+self.batch_size]

                batch_features = [
                    self.build_sequence(data) for data in batch_data
                ]

                x = torch.from_numpy(np.array(batch_features)).float().to(self.device)
                y = torch.tensor(batch_labels, dtype=torch.long).to(self.device)

                out = self.model(x)
                loss = self.criterion(out, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            train_loss = total_loss / int(np.ceil(len(events_train) / self.batch_size))

            # -------- VALIDATION --------
            self.model.eval()
            val_loss = 0

            with torch.no_grad():
                for i in range(0, len(events_val), self.batch_size):

                    batch_data = events_val[i:i+self.batch_size]
                    batch_labels = labels_val[i:i+self.batch_size]

                    batch_features = [
                        self.build_sequence(data) for data in batch_data
                    ]

                    x = torch.from_numpy(np.array(batch_features)).float().to(self.device)
                    y = torch.tensor(batch_labels, dtype=torch.long).to(self.device)

                    out = self.model(x)
                    loss = self.criterion(out, y)

                    val_loss += loss.item()

            val_loss = val_loss / int(np.ceil(len(events_val) / self.batch_size))

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

                batch_data = events[i:i+self.batch_size]

                batch_features = [
                    self.build_sequence(data) for data in batch_data
                ]

                x = torch.from_numpy(np.array(batch_features)).float().to(self.device)

                out = self.model(x)
                preds.extend(torch.argmax(out, dim=1).cpu().numpy())

        return np.array(preds)