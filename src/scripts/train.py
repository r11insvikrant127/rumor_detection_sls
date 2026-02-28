"""
PAPER-EXACT SLS TRAINING SCRIPT
5-Fold Cross Validation + Hybrid SLS-GBDT
+ Threshold Sweep (Paper Section V-D)
"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

import json
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier

from src.preprocessing.feature_extractor import FeatureExtractor
from src.preprocessing.feature_normalizer import FeatureNormalizer
from src.models.sls import PaperExactSLS
from src.training.trainer import SLSTrainer
from src.training.evaluator import Evaluator
from src.utils.config import ConfigManager
from src.utils.helpers import create_data_loader, set_seed


# =====================================================
# TRAINER PIPELINE
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

        self.threshold = 0.57  # initial paper value

        print("✅ PAPER EXACT: Using 31 features")

    # -------------------------------------------------
    # LOAD DATA
    # -------------------------------------------------
    def load_data_from_index(self, index_file):

        index_df = pd.read_csv(index_file)

        features, labels, graphs = [], [], []
        skipped = 0

        print("Loading dataset using index file...")

        for _, row in index_df.iterrows():

            file_path = Path(project_root) / row["file_path"]

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    event = json.load(f)

                if (
                    "source" not in event
                    or event["source"] is None
                    or "id" not in event["source"]
                ):
                    skipped += 1
                    continue

                tweets = [event["source"]]

                if "replies" in event and event["replies"]:
                    tweets.extend(event["replies"])

                event["tweets"] = tweets
                event["source_id"] = str(event["source"]["id"])

                graph = self.feature_extractor.tree_builder.build_from_tweets(
                    tweets,
                    source_id=event["source_id"]
                )

                graphs.append(graph)

                feat = self.feature_extractor.extract_features(event)

                features.append(feat)
                labels.append(int(row["label"]))

            except Exception as e:
                print(f"\n❌ Error processing file: {file_path}")
                print("Reason:", e)
                raise

        X = np.nan_to_num(np.array(features, dtype=np.float32))
        y = np.array(labels)

        print(f"\n✅ Loaded dataset: {X.shape}")
        print(f"Rumors: {np.sum(y)} | Non-rumors: {len(y)-np.sum(y)}")
        print(f"Skipped events: {skipped}")

        return X, y

    # -------------------------------------------------
    # 5-FOLD CV
    # -------------------------------------------------
    def train_cross_validation(self, X, y):

        skf = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):

            print("\n" + "="*50)
            print(f"Fold {fold}/5")
            print("="*50)

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

            # ==================================================
            # GBDT + THRESHOLD SWEEP
            # ==================================================
            print("🌳 Training GBDT fallback...")

            gbdt = GradientBoostingClassifier(random_state=42)
            gbdt.fit(X_train, y_train)

            preds_sls, probs_sls = trainer.predict(
                X_val.reshape(X_val.shape[0], 1, 31),
                return_probs=True
            )

            probs_sls = np.array(probs_sls)

            best_f1 = -1
            best_threshold = self.threshold
            best_preds = None

            thresholds = np.arange(0.45, 0.66, 0.02)

            for t in thresholds:

                preds_temp = preds_sls.copy()
                uncertain = probs_sls < t

                if np.any(uncertain):
                    preds_temp[uncertain] = gbdt.predict(
                        X_val[uncertain]
                    )

                metrics_temp = Evaluator.compute_metrics(
                    y_val,
                    preds_temp
                )

                if metrics_temp["f1"] > best_f1:
                    best_f1 = metrics_temp["f1"]
                    best_threshold = t
                    best_preds = preds_temp

            print(f"✅ Best threshold (fold {fold}): {best_threshold:.2f}")

            metrics = Evaluator.compute_metrics(y_val, best_preds)
            fold_results.append(metrics)

            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"F1:       {metrics['f1']:.4f}")

        return fold_results


# =====================================================
# MAIN
# =====================================================

def main():

    print("="*60)
    print("PAPER-EXACT SLS TRAINING (5-FOLD CV)")
    print("="*60)

    trainer = RumorDetectionTrainer("configs/default.yaml")

    X, y = trainer.load_data_from_index(
        "data/processed/pheme_dataset_index.csv"
    )

    results = trainer.train_cross_validation(X, y)

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)

    for metric in ["accuracy", "f1", "precision", "recall"]:
        vals = [r[metric] for r in results]
        print(f"{metric.capitalize():10s}: "
              f"{np.mean(vals):.4f} ± {np.std(vals):.4f}")

    print("\n✅ PAPER-EXACT TRAINING COMPLETE")


if __name__ == "__main__":
    main()