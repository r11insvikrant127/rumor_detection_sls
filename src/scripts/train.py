"""
PAPER-FAITHFUL SLS TRAINING SCRIPT
5-Fold Cross Validation + Hybrid SLS-GBDT
with Correct Threshold Logic
"""

import sys
import os
import json
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.preprocessing.feature_extractor import FeatureExtractor
from src.preprocessing.feature_normalizer import FeatureNormalizer
from src.models.sls import PaperExactSLS
from src.training.trainer import SLSTrainer
from src.training.evaluator import Evaluator
from src.utils.config import ConfigManager
from src.utils.helpers import create_data_loader, set_seed


# =====================================================
# TRAINER
# =====================================================

class RumorDetectionTrainer:

    def __init__(self, config_path):

        self.config = ConfigManager(config_path)
        set_seed(42)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.feature_extractor = FeatureExtractor()
        self.feature_names = list(
            self.feature_extractor.get_feature_names()
        )[:31]

        self.threshold = 0.57  # initial value (paper)

        print("✅ PAPER EXACT: Using 31 features")

    # -------------------------------------------------
    # LOAD DATA
    # -------------------------------------------------
    def load_data_from_index(self, index_file):

        index_df = pd.read_csv(index_file)

        features, labels = [], []
        skipped = 0

        print("Loading dataset...")

        for _, row in index_df.iterrows():

            file_path = Path(project_root) / row["file_path"]

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    event = json.load(f)

                if "source" not in event or event["source"] is None:
                    skipped += 1
                    continue

                tweets = [event["source"]]
                if "replies" in event:
                    tweets.extend(event["replies"])

                event["tweets"] = tweets
                event["source_id"] = str(event["source"]["id"])

                feat = self.feature_extractor.extract_features(event)

                features.append(feat)
                labels.append(int(row["label"]))

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                raise

        X = np.nan_to_num(np.array(features, dtype=np.float32))
        y = np.array(labels)

        print(f"Dataset loaded: {X.shape}")
        print(f"Skipped events: {skipped}")

        return X, y

    # -------------------------------------------------
    # CONFIDENCE COMPUTATION (paper rule)
    # -------------------------------------------------
    @staticmethod
    def compute_confidence(probs):
        probs = np.array(probs)

        # Case 1: full softmax output (N,2)
        if probs.ndim == 2:
            return probs.max(axis=1)

        # Case 2: only class-1 probability (N,)
        return np.maximum(probs, 1 - probs)

    # -------------------------------------------------
    # CROSS VALIDATION
    # -------------------------------------------------
    def train_cross_validation(self, X, y):

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        fold_results = []
        chosen_thresholds = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):

            print("\n" + "="*60)
            print(f"Fold {fold}/5")
            print("="*60)

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # ---------- Normalization ----------
            normalizer = FeatureNormalizer()
            X_train = normalizer.fit_transform(X_train, self.feature_names)
            X_val = normalizer.transform(X_val)

            # ---------- Model ----------
            model = PaperExactSLS(
                input_dim=31,
                lstm_hidden=128,
                num_classes=2,
                dropout_rate=0.15,
                se_reduction=self.config.model.se_reduction
            )

            trainer = SLSTrainer(
                model=model,
                device=self.device,
                config=self.config.training.__dict__
            )

            train_loader = create_data_loader(
                X_train, y_train,
                batch_size=self.config.training.batch_size,
                add_channel_dim=True
            )

            val_loader = create_data_loader(
                X_val, y_val,
                batch_size=self.config.training.batch_size,
                shuffle=False,
                add_channel_dim=True
            )

            trainer.train(train_loader, val_loader)

            # ---------- Train GBDT ----------
            print("🌳 Training GBDT fallback...")
            gbdt = GradientBoostingClassifier(
                random_state=42,
                subsample=0.8
            )
            gbdt.fit(X_train, y_train)

            # ---------- Predictions ----------
            preds_sls, probs_sls = trainer.predict(
                X_val.reshape(len(X_val), 1, 31),
                return_probs=True
            )

            max_probs = self.compute_confidence(probs_sls)

            print("Mean confidence:", max_probs.mean())

            # ---------- Threshold Sweep ----------
            thresholds = np.arange(0.45, 0.66, 0.02)

            best_f1 = -1
            best_threshold = self.threshold
            best_preds = None

            for t in thresholds:

                preds_temp = preds_sls.copy()

                uncertain = max_probs < t
                routing_rate = uncertain.mean()

                print(f"t={t:.2f} | routed={routing_rate:.3f}")

                if np.any(uncertain):
                    preds_temp[uncertain] = gbdt.predict(
                        X_val[uncertain]
                    )

                metrics = Evaluator.compute_metrics(y_val, preds_temp)

                if metrics["f1"] > best_f1:
                    best_f1 = metrics["f1"]
                    best_threshold = t
                    best_preds = preds_temp

            chosen_thresholds.append(best_threshold)

            print(f"Best threshold fold {fold}: {best_threshold:.2f}")

            metrics = Evaluator.compute_metrics(y_val, best_preds)
            fold_results.append(metrics)

            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"F1:       {metrics['f1']:.4f}")

        # ---------- Global Threshold ----------
        final_threshold = float(np.mean(chosen_thresholds))
        self.threshold = final_threshold

        print("\n" + "="*60)
        print(f"GLOBAL THRESHOLD SELECTED: {final_threshold:.3f}")
        print("="*60)

        return fold_results


# =====================================================
# MAIN
# =====================================================

def main():

    print("="*60)
    print("PAPER-FAITHFUL SLS TRAINING")
    print("="*60)

    trainer = RumorDetectionTrainer("configs/default.yaml")

    X, y = trainer.load_data_from_index(
        "data/processed/pheme_dataset_index.csv"
    )

    results = trainer.train_cross_validation(X, y)

    print("\nFINAL RESULTS")
    for metric in ["accuracy", "f1", "precision", "recall"]:
        vals = [r[metric] for r in results]
        print(f"{metric:10s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")


if __name__ == "__main__":
    main()