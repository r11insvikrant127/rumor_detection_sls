"""
PAPER-EXACT SLS TRAINING SCRIPT
5-Fold Cross Validation + Hybrid SLS-GBDT
(FULLY PAPER-FAITHFUL VERSION)
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

        # Paper uses 31 features
        self.feature_names = list(
            self.feature_extractor.get_feature_names()
        )[:31]

        self.threshold = 0.57  # paper value

        print("✅ PAPER EXACT: Using 31 features")

    # -------------------------------------------------
    # LOAD DATA FROM INDEX
    # -------------------------------------------------
    def load_data_from_index(self, index_file):

        index_df = pd.read_csv(index_file)

        features, labels, graphs = [], [], []

        print("Loading dataset using index file...")

        skipped = 0

        for _, row in index_df.iterrows():

            file_path = Path(project_root) / row["file_path"]

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    event = json.load(f)

                # =========================================
                # PAPER REQUIREMENT:
                # each event MUST have one source tweet
                # =========================================
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

                if len(tweets) == 0:
                    skipped += 1
                    continue

                # -------------------------------------------------
                # IMPORTANT FIX (PAPER ALIGNMENT)
                # -------------------------------------------------
                event["tweets"] = tweets
                event["source_id"] = str(event["source"]["id"])

                # -------------------------------------------------
                # BUILD PROPAGATION GRAPH (SANITY CHECK)
                # -------------------------------------------------
                try:
                    graph = self.feature_extractor.tree_builder.build_from_tweets(
                        tweets,
                        source_id=event["source_id"]
                    )
                except Exception:
                    skipped += 1
                    continue

                graphs.append(graph)

                # -------------------------------------------------
                # FEATURE EXTRACTION
                # -------------------------------------------------
                feat = self.feature_extractor.extract_features(event)

                features.append(feat)
                labels.append(int(row["label"]))

            except Exception as e:
                print(f"\n❌ Error processing file: {file_path}")
                print("Reason:", e)
                raise

        X = np.array(features, dtype=np.float32)
        y = np.array(labels)

        X = np.nan_to_num(X)

        print(f"\n✅ Loaded dataset: {X.shape}")
        print(f"Rumors: {np.sum(y)} | Non-rumors: {len(y)-np.sum(y)}")
        print(f"Skipped events: {skipped}")

        # =====================================================
        # PROPAGATION SANITY CHECK
        # =====================================================
        print("\n🔎 PROPAGATION SANITY CHECK")

        if len(graphs) > 0:
            avg_nodes = np.mean([g.number_of_nodes() for g in graphs])
            avg_depth = np.mean([
                self.feature_extractor.tree_builder
                .get_tree_metrics(g)["max_depth"]
                for g in graphs
            ])

            print(f"Avg nodes/event : {avg_nodes:.2f}")
            print(f"Avg max depth   : {avg_depth:.2f}")
        else:
            print("⚠ No graphs available for sanity check.")

        return X, y

    # -------------------------------------------------
    # 5-FOLD CV (PAPER PROTOCOL)
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
            
            # =====================================================
            # DEBUG 1 — RAW FEATURE STATS (before normalization)
            # =====================================================
            if fold == 1:   # print once only
                print("\n[DEBUG] RAW FEATURES")
                print("Train mean :", np.mean(X_train))
                print("Train std  :", np.std(X_train))
                print("Train min  :", np.min(X_train))
                print("Train max  :", np.max(X_train))

            # -------- Normalization (Eq.2 paper) --------
            normalizer = FeatureNormalizer()
            X_train = normalizer.fit_transform(X_train, self.feature_names)
            X_val = normalizer.transform(X_val)
            
            # =====================================================
            # DEBUG 2 — AFTER NORMALIZATION (CRITICAL)
            # =====================================================
            if fold == 1:
                print("\n[DEBUG] NORMALIZED FEATURES")
                print("Train mean :", np.mean(X_train))
                print("Train std  :", np.std(X_train))
                print("Train min  :", np.min(X_train))
                print("Train max  :", np.max(X_train))

            # -------- SLS Model --------
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
            
            # =====================================================
            # DEBUG 3 — BATCH ENTERING MODEL
            # =====================================================
            if fold == 1:
                xb, yb = next(iter(train_loader))
                print("\n[DEBUG] MODEL INPUT BATCH")
                print("Tensor mean :", xb.mean().item())
                print("Tensor std  :", xb.std().item())
                print("Tensor shape:", xb.shape)

            trainer.train(train_loader, val_loader)

            # -------- GBDT fallback (paper Section IV-F) --------
            print("🌳 Training GBDT fallback...")
            gbdt = GradientBoostingClassifier(random_state=42)
            gbdt.fit(X_train, y_train)

            preds_sls, probs_sls = trainer.predict(
                X_val.reshape(X_val.shape[0], 1, 31),
                return_probs=True
            )

            preds_final = preds_sls.copy()
            probs_sls = np.array(probs_sls)

            uncertain = probs_sls < self.threshold

            if np.any(uncertain):
                preds_final[uncertain] = gbdt.predict(X_val[uncertain])

            metrics = Evaluator.compute_metrics(y_val, preds_final)
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