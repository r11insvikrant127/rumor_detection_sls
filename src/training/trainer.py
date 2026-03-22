import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from copy import deepcopy

from .loss import CircleLoss
from .evaluator import Evaluator


class SLSTrainer:

    # =====================================================
    # INIT
    # =====================================================
    def __init__(self, model, device="cuda", config=None):

        self.model = model.to(device)
        self.device = device
        self.config = config or {}

        # ---- Circle Loss ----
        self.criterion = CircleLoss(m=0.25, gamma=256)

        # ---- Optimizer ----
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.config.get("learning_rate", 1e-3)),
            betas=(0.9, 0.999),
            weight_decay=float(self.config.get("weight_decay", 1e-4)),
        )

        # ---- Scheduler ----
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=float(self.config.get("scheduler_factor", 0.5)),
            patience=int(self.config.get("scheduler_patience", 5)),
        )

        self.epochs = int(self.config.get("epochs", 100))

        # ✅ NEW: temperature scaling (inference only)
        self.temperature = float(self.config.get("temperature", 1.0))

        # ✅ NEW: early stopping
        self.early_stop = self.config.get("early_stopping", True)
        self.patience = int(self.config.get("early_stopping_patience", 10))

    # =====================================================
    # PREDICT (Eq.9 + temperature scaling)
    # =====================================================
    def predict(self, X, return_probs=False):

        was_training = self.model.training   # 🔥 store state

        self.model.eval()

        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32)

        X = X.to(self.device)

        with torch.no_grad():
            outputs = self.model(X)
            probs = torch.softmax(outputs / self.temperature, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds = preds.cpu().numpy()

        if was_training:
            self.model.train()   # restore only if needed

        if return_probs:
            return all_preds, probs.cpu().numpy()

        return all_preds

    # =====================================================
    # TRAIN ONE EPOCH
    # =====================================================
    def train_epoch(self, loader, epoch):

        self.model.train()

        total_loss = 0
        all_preds, all_labels = [], []

        pbar = tqdm(loader, desc=f"Epoch {epoch}")

        for features, labels in pbar:

            features = features.to(self.device)
            labels = labels.long().to(self.device)

            outputs = self.model(features)

            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                float(self.config.get("grad_clip", 1.0)),
            )

            self.optimizer.step()

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            batch_acc = (preds == labels).float().mean().item()
            pbar.set_postfix(loss=loss.item(), acc=f"{batch_acc:.2%}")

        epoch_loss = total_loss / len(loader)
        epoch_acc = (np.array(all_preds) == np.array(all_labels)).mean()
        epoch_f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)

        return epoch_loss, epoch_acc, epoch_f1

    # =====================================================
    # VALIDATION
    # =====================================================
    def validate(self, loader):

        self.model.eval()

        all_preds, all_labels = [], []
        total_loss = 0

        with torch.no_grad():
            for features, labels in loader:

                features = features.to(self.device)
                labels = labels.long().to(self.device)

                outputs = self.model(features)

                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                

        metrics = Evaluator.compute_metrics(
            np.array(all_labels),
            np.array(all_preds)
        )

        metrics["val_loss"] = total_loss / max(len(loader), 1)

        return metrics

    # =====================================================
    # MAIN TRAIN LOOP
    # =====================================================
    def train(self, train_loader, val_loader):

        print("=" * 60)
        print("PAPER-FAITHFUL TRAINING")
        print("=" * 60)

        best_f1 = -1
        best_state = None
        no_improve = 0
        best_metrics = None
        for epoch in range(1, self.epochs + 1):

            train_loss, train_acc, train_f1 = self.train_epoch(
                train_loader, epoch
            )

            val_metrics = self.validate(val_loader)

            self.scheduler.step(val_metrics["f1"])

            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss {train_loss:.4f} | "
                f"Train Acc {train_acc:.4f} | "
                f"Train F1 {train_f1:.4f} | "
                f"Val F1 {val_metrics['f1']:.4f}"
            )

            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                best_state = deepcopy(self.model.state_dict())
                best_metrics = val_metrics
                no_improve = 0
            else:
                no_improve += 1

            # ✅ Early stopping
            if self.early_stop and no_improve >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return best_f1, best_metrics