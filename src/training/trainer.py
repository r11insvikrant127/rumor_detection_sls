"""
Paper-faithful trainer for SLS model.
Implements training protocol exactly as described in the paper.

Key Principles:
---------------
TRAINING:
    - Circle Loss operates on cosine similarities
    - NO softmax during optimization

INFERENCE (Paper Eq.9):
    - Softmax applied to cosine outputs
    - Threshold applied on probabilities
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

    Paper settings:
        - Circle Loss (m=0.25, gamma=256)
        - Adam optimizer
        - Fixed 100 epochs
        - No scheduler
        - No early stopping
        - Final epoch model used
    """

    # =====================================================
    # INIT
    # =====================================================
    def __init__(self, model, device="cuda", config=None):

        self.model = model.to(device)
        self.device = device
        self.config = config or {}

        # Paper loss
        self.criterion = CircleLoss(m=0.25, gamma=256)

        # Paper optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.get("learning_rate", 1e-3),
            betas=(0.9, 0.999),
            weight_decay=0.0
        )

        self.epochs = self.config.get("epochs", 100)

        # Paper confidence threshold
        self.threshold = 0.57

    # =====================================================
    # PREDICT (Inference — PAPER Eq.9)
    # =====================================================
    def predict(self, X, return_probs=False):

        self.model.eval()

        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)

        X = X.to(self.device)

        with torch.no_grad():

            outputs = self.model(X)

            # PAPER Eq.(9):
            # y_hat = Softmax(FC(S))
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

            # Forward pass
            outputs = self.model(features)

            # Circle Loss on cosine similarities
            loss = self.criterion(outputs, labels)

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Prediction rule (cosine classifier)
            preds = torch.argmax(outputs, dim=1)

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

                # Softmax ONLY for evaluation metrics
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        metrics = Evaluator.compute_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs)
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

        # Paper uses FINAL epoch model
        return final_metrics["f1"], final_metrics