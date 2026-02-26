"""
Paper-faithful trainer for SLS model.
Implements training protocol exactly as described in the paper.
"""

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score

from .loss import CircleLoss
from .evaluator import Evaluator


class SLSTrainer:
    """
    Paper-Exact SLS Trainer.

    Paper training settings:
        - Circle Loss (m=0.25, gamma=256)
        - Adam optimizer
        - Fixed epochs (100)
        - No scheduler
        - No early stopping
        - Final epoch model used
    """

    def __init__(self, model, device="cuda", config=None):

        self.model = model.to(device)
        self.device = device
        self.config = config or {}

        # -------------------------
        # PAPER LOSS
        # -------------------------
        self.criterion = CircleLoss(m=0.25, gamma=256)

        # -------------------------
        # PAPER OPTIMIZER
        # -------------------------
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.get("learning_rate", 1e-3),
            betas=(0.9, 0.999),
            weight_decay=0.0
        )

        # Paper uses fixed epochs
        self.epochs = self.config.get("epochs", 100)

        # Paper threshold (used later in inference)
        self.threshold = 0.57
    
    # =====================================================
    # PREDICT (Inference)
    # =====================================================
    def predict(self, X, return_probs=False):
        """
        Run inference using trained SLS model.

        Args:
            X : numpy array or tensor
                shape (N, 1, 31) or (N, 31)
            return_probs : bool

        Returns:
            predictions, probabilities(optional)
        """

        self.model.eval()

        # Convert numpy → tensor
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)

        X = X.to(self.device)

        all_preds = []
        all_probs = []

        with torch.no_grad():

            outputs = self.model(X)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds = preds.cpu().numpy()
            all_probs = probs[:, 1].cpu().numpy()

        if return_probs:
            return all_preds, all_probs

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

            # Forward
            outputs = self.model(features)

            loss = self.criterion(outputs, labels)

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Metrics
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            batch_acc = (preds == labels).float().mean().item()
            pbar.set_postfix(loss=loss.item(), acc=f"{batch_acc:.2%}")

        epoch_loss = total_loss / len(loader)
        epoch_acc = (np.array(all_preds) == np.array(all_labels)).mean()

        epoch_f1 = f1_score(all_labels, all_preds, average="binary")

        return epoch_loss, epoch_acc, epoch_f1

    # =====================================================
    # VALIDATION
    # =====================================================
    def validate(self, loader):

        self.model.eval()

        all_preds, all_labels, all_probs = [], [], []
        total_loss = 0

        with torch.no_grad():
            for features, labels in loader:

                features = features.to(self.device)
                labels = labels.long().to(self.device)

                outputs = self.model(features)

                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        metrics = Evaluator.compute_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs)[:, 1]
        )

        metrics["val_loss"] = total_loss / len(loader)

        return metrics

    # =====================================================
    # MAIN TRAIN LOOP
    # =====================================================
    def train(self, train_loader, val_loader):

        print("=" * 60)
        print("PAPER-EXACT TRAINING")
        print("=" * 60)
        print("Loss: Circle Loss (γ=256, m=0.25)")
        print("Optimizer: Adam")
        print("Scheduler: None")
        print("Early stopping: Disabled")
        print(f"Epochs: {self.epochs}")
        print("=" * 60)

        final_metrics = None

        for epoch in range(1, self.epochs + 1):

            train_loss, train_acc, train_f1 = self.train_epoch(
                train_loader, epoch
            )

            val_metrics = self.validate(val_loader)
            final_metrics = val_metrics

            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss {train_loss:.4f} | "
                f"Val F1 {val_metrics['f1']:.4f}"
            )

        # PAPER: use FINAL epoch model
        return final_metrics["f1"], final_metrics