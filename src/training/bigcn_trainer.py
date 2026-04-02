import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score


class BiGCNTrainer:

    def __init__(self, model, device="cuda", lr=5e-4, epochs=200, config=None):
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs

        # ✅ SAME LR (paper-faithful)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=1e-4
        )

        self.early_stop = True if config is None else config.get("early_stopping", True)
        self.patience = 10 if config is None else config.get("early_stopping_patience", 10)

    def train(self, train_dataset, val_dataset, batch_size=64):

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # ===== class weights =====
        labels = [int(d.y.item()) for d in train_dataset]
        class_counts = np.bincount(labels)
        weights = 1.0 / (class_counts + 1e-6)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

        criterion = nn.CrossEntropyLoss(weight=weights)

        best_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(self.epochs):

            # ================= TRAIN =================
            self.model.train()
            total_loss = 0

            for data in tqdm(train_loader, desc=f"Epoch {epoch+1}"):

                data = data.to(self.device)

                # ✅ NO DropEdge here (handled in model)
                out = self.model(data)

                loss = criterion(out, data.y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            train_loss = total_loss / len(train_loader)

            # ================= VALIDATION =================
            self.model.eval()
            val_loss = 0
            correct = 0
            total = 0

            all_preds = []
            all_labels = []

            with torch.no_grad():
                for data in val_loader:

                    data = data.to(self.device)
                    out = self.model(data)

                    loss = criterion(out, data.y)
                    val_loss += loss.item()

                    pred = out.argmax(dim=1)

                    correct += (pred == data.y).sum().item()
                    total += len(data.y)

                    all_preds.extend(pred.cpu().numpy())
                    all_labels.extend(data.y.cpu().numpy())

            val_loss /= len(val_loader)
            val_acc = correct / total
            val_f1 = f1_score(all_labels, all_preds, average="macro")

            print(f"[BiGCN] Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

            # ================= EARLY STOP =================
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = self.model.state_dict()
                no_improve = 0
            else:
                no_improve += 1

            if self.early_stop and no_improve >= self.patience:
                print(f"[BiGCN] Early stopping at epoch {epoch+1}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def predict(self, dataset, batch_size=64):

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()
        preds = []

        with torch.no_grad():
            for data in loader:

                data = data.to(self.device)
                out = self.model(data)

                pred = out.argmax(dim=1)
                preds.extend(pred.cpu().numpy())

        return np.array(preds)