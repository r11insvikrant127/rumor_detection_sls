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

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from src.models.bigcn import BiGCN
from src.models.rvnn import RvNN
from src.models.ppc import PPC
from src.preprocessing.tree_builder_ppc import TreeBuilderPPC
from src.training.bigcn_trainer import BiGCNTrainer
from src.training.rvnn_trainer import RvNNTrainer
from src.training.ppc_trainer import PPCTrainer
from src.preprocessing import fit_tfidf_bigcn


from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

# =====================================================
# TABLE III PRINT
# =====================================================
def print_full_table3(all_results):

    print("\n" + "="*70)
    print("FULL TABLE III (ALL MODELS)")
    print("="*70)

    order = [
        "DTC",
        "SVM-RBF",
        "SVM-TS",
        "PPC",
        "RvNN",
        "BiGCN",
        "GBDT",
        "SLS-",
        "SLS"
    ]

    print(f"{'Method':<12} {'Class':<5} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8}")
    print("-"*70)

    for method in order:

        results = all_results[method]

        acc = np.mean([r["accuracy"] for r in results])

        prec_n = np.mean([r["precision_non_rumor"] for r in results])
        rec_n  = np.mean([r["recall_non_rumor"] for r in results])
        f1_n   = np.mean([r["f1_non_rumor"] for r in results])

        prec_r = np.mean([r["precision_rumor"] for r in results])
        rec_r  = np.mean([r["recall_rumor"] for r in results])
        f1_r   = np.mean([r["f1_rumor"] for r in results])

        print(f"{method:<12} {'N':<5} {acc:<8.4f} {prec_n:<8.4f} {rec_n:<8.4f} {f1_n:<8.4f}")
        print(f"{'':<12} {'R':<5} {'':<8} {prec_r:<8.4f} {rec_r:<8.4f} {f1_r:<8.4f}")


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

        self.threshold = self.config.gbdt.threshold

        print("✅ PAPER EXACT: Using 31 features")

    # -------------------------------------------------
    # LOAD DATA
    # -------------------------------------------------
    def load_data_from_index(self, index_file):

        index_df = pd.read_csv(index_file)

        features, labels, events = [], [], []

        print("Loading dataset...")

        for _, row in index_df.iterrows():

            file_path = Path(project_root) / row["file_path"]

            with open(file_path, "r", encoding="utf-8") as f:
                event = json.load(f)

            if "source" not in event or event["source"] is None:
                continue

            tweets = [event["source"]]
            if "replies" in event:
                tweets.extend(event["replies"])

            event["tweets"] = tweets
            event["source_id"] = str(event["source"]["id"])

            feat = self.feature_extractor.extract_features(event)

            features.append(feat)
            labels.append(int(row["label"]))
            events.append(event)

        X = np.nan_to_num(np.array(features, dtype=np.float32))
        y = np.array(labels)

        print(f"Dataset loaded: {X.shape}")

        return X, y, events


    # -------------------------------------------------
    # CROSS VALIDATION
    # -------------------------------------------------
    def train_cross_validation(self, X, y, events):

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        fold_results = []

        all_model_results = {
            "SLS": [],
            "SLS-": [],
            "GBDT": [],
            "DTC": [],
            "SVM-RBF": [],
            "SVM-TS": [],
            "BiGCN": [],
            "RvNN": [],
            "PPC": []
        }
        # =====================================================
        # GRAPH CACHING (BUILD ONCE)
        # =====================================================
        print("🔄 Building graph cache...")
        
        tree_builder = TreeBuilderPPC()
        graph_cache = []

        for event in events:
            graph = tree_builder.build_from_tweets(
                tweets=event["tweets"],
                source_id=event["source_id"]
            )
            graph_cache.append(graph)

        print("✅ Graph cache ready")
        
        # 🔥 FIT TF-IDF ON TRAINING DATA
        print("🔧 Fitting TF-IDF for BiGCN...")
        fit_tfidf_bigcn(graph_cache)

        # 🔥 DEBUG CHECK (ADD HERE)
        graph = graph_cache[0]

        print("\n🔍 PPC GRAPH CHECK:")
        for node, attr in graph.nodes(data=True):
            print(attr)
            break
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):

            print("\n" + "="*60)
            print(f"Fold {fold}/5")
            print("="*60)

            X_train_raw, X_val_raw = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            graphs_val = [graph_cache[i] for i in val_idx]

            # Normalize
            normalizer = FeatureNormalizer()
            X_train = normalizer.fit_transform(X_train_raw.copy(), self.feature_names)
            X_val = normalizer.transform(X_val_raw.copy())
            # -------- SVM Scaling --------
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_raw)
            X_val_scaled = scaler.transform(X_val_raw)

            # -------- SLS --------
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
                add_channel_dim=True,
                use_weighted_sampler=True,
            )

            val_loader = create_data_loader(
                X_val, y_val,
                batch_size=self.config.training.batch_size,
                shuffle=False,
                add_channel_dim=True
            )

            trainer.train(train_loader, val_loader)

            # =====================================================
            # TRAIN DEEP MODELS (BiGCN, RvNN, PPC)
            # =====================================================
            graphs_train = [graph_cache[i] for i in train_idx]
            graphs_val = [graph_cache[i] for i in val_idx]

            # ---- BiGCN ----
            bigcn_trainer = BiGCNTrainer(
                BiGCN(in_dim=3000),
                device=self.device,
                config=self.config.training.__dict__
            )
            bigcn_trainer.train(graphs_train, y_train, graphs_val, y_val)
            preds_bigcn = bigcn_trainer.predict(graphs_val)

            # ---- RvNN ----
            rvnn_trainer = RvNNTrainer(
                RvNN(),
                device=self.device,
                config=self.config.training.__dict__
            )
            rvnn_trainer.train(graphs_train, y_train, graphs_val, y_val)
            preds_rvnn = rvnn_trainer.predict(graphs_val)

            # ---- PPC ----
            ppc_trainer = PPCTrainer(
                PPC(input_dim=8),
                device=self.device,
                config=self.config.training.__dict__
            )
            ppc_trainer.train(graphs_train, y_train, graphs_val, y_val)
            preds_ppc = ppc_trainer.predict(graphs_val)

            # ---- store metrics ----
            all_model_results["BiGCN"].append(
                Evaluator.compute_metrics(y_val, preds_bigcn)
            )

            all_model_results["RvNN"].append(
                Evaluator.compute_metrics(y_val, preds_rvnn)
            )

            all_model_results["PPC"].append(
                Evaluator.compute_metrics(y_val, preds_ppc)
            )

            # -------- GBDT --------
            print("🌳 Training GBDT...")
            gbdt = GradientBoostingClassifier(random_state=42)
            gbdt.fit(X_train_raw, y_train)

            print("🔮 SLS inference...")
            preds_sls, probs_sls = trainer.predict(X_val, return_probs=True)
            max_probs = probs_sls.max(axis=1)

            preds_temp = preds_sls.copy()
            uncertain = max_probs < self.threshold

            if np.any(uncertain):
                preds_temp[uncertain] = gbdt.predict(X_val_raw[uncertain])

            # SLS
            metrics = Evaluator.compute_metrics(y_val, preds_temp)
            fold_results.append(metrics)
            all_model_results["SLS"].append(metrics)

            # SLS-
            all_model_results["SLS-"].append(
                Evaluator.compute_metrics(y_val, preds_sls)
            )

            # GBDT
            preds_gbdt = gbdt.predict(X_val_raw)
            all_model_results["GBDT"].append(
                Evaluator.compute_metrics(y_val, preds_gbdt)
            )

            # -------- Classical --------
            print("🌲 Training Decision Tree...")
            dtc = DecisionTreeClassifier(random_state=42)
            dtc.fit(X_train_raw, y_train)
            all_model_results["DTC"].append(
                Evaluator.compute_metrics(y_val, dtc.predict(X_val_raw))
            )

            print("⚡ Training SVM-RBF...")
            svm_rbf = SVC(kernel="rbf", max_iter=10000)
            svm_rbf.fit(X_train_scaled, y_train)

            all_model_results["SVM-RBF"].append(
                Evaluator.compute_metrics(y_val, svm_rbf.predict(X_val_scaled))
            )

            print("⚡ Training SVM-TS...")
            svm_ts = LinearSVC(max_iter=10000)
            svm_ts.fit(X_train_scaled, y_train)

            all_model_results["SVM-TS"].append(
                Evaluator.compute_metrics(y_val, svm_ts.predict(X_val_scaled))
            )

        return fold_results, all_model_results


# =====================================================
# MAIN
# =====================================================
def main():

    trainer = RumorDetectionTrainer("configs/default.yaml")

    X, y, events = trainer.load_data_from_index(
        "data/processed/pheme_dataset_index.csv"
    )

    results, all_results = trainer.train_cross_validation(X, y, events)

    print("\nFINAL RESULTS")
    for metric in ["accuracy", "f1"]:
        vals = [r[metric] for r in results]
        print(f"{metric}: {np.mean(vals):.4f}")

    print_full_table3(all_results)


if __name__ == "__main__":
    main()