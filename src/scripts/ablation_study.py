"""
PAPER-EXACT ablation study for SLS reproduction.
Reproduces Section V-D, Fig.4 and Fig.5 exactly.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import torch
import json
import warnings
warnings.filterwarnings("ignore")

from src.preprocessing.feature_extractor import FeatureExtractor
from src.preprocessing.feature_normalizer import FeatureNormalizer
from src.models.sls import PaperExactSLS
from src.models.gbdt_wrapper import GBDTWrapper
from src.training.trainer import SLSTrainer
from src.utils.config import ConfigManager
from src.utils.helpers import create_data_loader, set_seed


class PaperExactAblation:

    def __init__(self, config_path):

        self.config = ConfigManager(config_path)
        set_seed(42)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.feature_extractor = FeatureExtractor()
        all_features = self.feature_extractor.get_feature_names()

        # Paper uses exactly 31 features
        self.feature_names = all_features[:31]

        # Feature groups (Table I)
        self.feature_groups = {
            "propagation": list(range(0,6)),
            "user": list(range(6,19)),
            "content": list(range(19,31))
        }

        self.thresholds_fig4 = np.round(np.arange(0.52,0.66,0.01),2)
        self.thresholds_fig5 = np.round(np.arange(0.55,0.81,0.01),2)

        self.results = {}

    # -------------------------------------------------------
    # DATA LOADING
    # -------------------------------------------------------

    def load_pheme_data(self, data_dir):

        index_file = Path(data_dir) / "pheme_dataset_index.csv"
        df = pd.read_csv(index_file)

        events = []

        for _, row in df.iterrows():

            try:
                thread_path = Path(row["file_path"])

                if not thread_path.exists():
                    continue

                with open(thread_path, encoding="utf-8") as f:
                    thread = json.load(f)

                # Skip if source missing
                if "source" not in thread or thread["source"] is None:
                    continue

                # EXACT format used in train.py
                tweets = [thread["source"]]

                if "replies" in thread:
                    tweets.extend(thread["replies"])

                thread["tweets"] = tweets
                thread["source_id"] = str(thread["source"]["id"])
                thread["label"] = int(row["label"])

                events.append(thread)

            except Exception as err:
                print("Skipping:", err)

        print("Loaded events:", len(events))

        return events

    # -------------------------------------------------------
    # FEATURE EXTRACTION
    # -------------------------------------------------------

    def extract_features(self,events):

        X=[]
        y=[]

        for e in events:

            try:

                feats = self.feature_extractor.extract_features(e)

                vec = feats[:31]

                X.append(vec)
                y.append(e["label"])

            except Exception as err:
                print("Feature extraction failed:", err)
                continue

        X=np.array(X,dtype=np.float32)
        y=np.array(y)
        if len(X) == 0:
            raise ValueError("No features extracted. FeatureExtractor is failing for all events.")

        X = np.nan_to_num(X, nan=0.0)

        print("Feature matrix shape:",X.shape)

        return X,y


    # -------------------------------------------------------
    # CROSS VALIDATION
    # -------------------------------------------------------

    def setup_cv(self,X,y):

        skf=StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

        self.splits=list(skf.split(X,y))


    # -------------------------------------------------------
    # TRAIN SLS MODEL
    # -------------------------------------------------------

    def train_sls(self,X_train,y_train,X_val,y_val,input_dim):

        model=PaperExactSLS(
            input_dim=input_dim,
            lstm_hidden=128,
            num_classes=2,
            dropout_rate=self.config.model.dropout_rate,
            se_reduction=self.config.model.se_reduction
        )

        trainer=SLSTrainer(
            model=model,
            device=self.device,
            config=self.config.training.__dict__
        )

        train_loader=create_data_loader(
            X_train,y_train,
            batch_size=self.config.training.batch_size,
            add_channel_dim=True
        )

        val_loader=create_data_loader(
            X_val,y_val,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            add_channel_dim=True
        )

        trainer.train(train_loader, val_loader)

        with torch.no_grad():

            _, probs = trainer.predict(X_val, return_probs=True)

        return probs
    
    # -------------------------------------------------------
    # FIGURE 4
    # -------------------------------------------------------

    def compute_fig4(self):

        print("\nRunning Fig4 reproduction")

        models = {
            "pSLS": self.feature_groups["propagation"],
            "uSLS": self.feature_groups["user"],
            "cSLS": self.feature_groups["content"],
            "SLS": list(range(31))
        }

        results = {}

        for name, idx in models.items():

            fold_probs = []
            fold_labels = []
            fold_features_train = []
            fold_features_val = []
            fold_labels_train = []

            # -----------------------------
            # TRAIN SLS + STORE FOLD DATA
            # -----------------------------
            for train_idx, val_idx in self.splits:

                # 1. Select subset
                X_train = self.X[train_idx][:, idx]
                y_train = self.y[train_idx]

                X_val = self.X[val_idx][:, idx]
                y_val = self.y[val_idx]

                # 2. Feature names
                selected_features = [self.feature_names[i] for i in idx]

                # 3. Normalize
                normalizer = FeatureNormalizer()
                X_train = normalizer.fit_transform(X_train, selected_features)
                X_val = normalizer.transform(X_val)

                # 4. Train SLS
                probs = self.train_sls(
                    X_train, y_train,
                    X_val, y_val,
                    len(idx)
                )

                # Store everything needed
                fold_probs.append(probs)
                fold_labels.append(y_val)
                fold_features_train.append(X_train)
                fold_features_val.append(X_val)
                fold_labels_train.append(y_train)

            # -----------------------------
            # THRESHOLD LOOP
            # -----------------------------
            threshold_acc = []

            for t in self.thresholds_fig4:

                fold_acc = []

                for probs, y_val, X_train, X_val, y_train in zip(
                    fold_probs, fold_labels,
                    fold_features_train, fold_features_val,
                    fold_labels_train
                ):

                    max_probs = np.max(probs, axis=1)
                    preds = np.argmax(probs, axis=1)

                    # --- Train GBDT (same fold, same features) ---
                    gbdt = GBDTWrapper(
                        n_estimators=self.config.gbdt.n_estimators,
                        learning_rate=self.config.gbdt.learning_rate,
                        max_depth=self.config.gbdt.max_depth
                    )
                    gbdt.fit(X_train, y_train)

                    # --- Apply switching ---
                    uncertain = max_probs < t

                    if uncertain.sum() > 0:
                        preds[uncertain] = gbdt.predict(X_val[uncertain])

                    acc = (preds == y_val).mean()
                    fold_acc.append(acc)

                # Mean over folds
                threshold_acc.append(np.nanmean(fold_acc))

            results[name] = threshold_acc

        self.results["fig4"] = results

    # -------------------------------------------------------
    # FIGURE 5
    # -------------------------------------------------------

    def compute_fig5(self):

        print("\nRunning Fig5 reproduction")

        rows = []

        for train_idx, val_idx in self.splits:

            # ---- 2. Get SLS features ----
            X_train = self.X[train_idx]
            y_train = self.y[train_idx]

            X_val = self.X[val_idx]
            y_val = self.y[val_idx]

            # ---- 3. Normalize for SLS (per fold) ----
            normalizer = FeatureNormalizer()
            X_train = normalizer.fit_transform(X_train, self.feature_names)
            X_val = normalizer.transform(X_val)

            # ---- 4. Train SLS ----
            probs = self.train_sls(X_train, y_train, X_val, y_val, 31)

            sls_preds = np.argmax(probs, axis=1)
            max_probs = np.max(probs, axis=1)

            # ---- 5. Train GBDT (RAW features) ----
            gbdt = GBDTWrapper(
                n_estimators=self.config.gbdt.n_estimators,
                learning_rate=self.config.gbdt.learning_rate,
                max_depth=self.config.gbdt.max_depth
            )

            gbdt.fit(X_train, y_train)
            gbdt_preds = gbdt.predict(X_val)

            # ---- 6. Threshold analysis ----
            for t in self.thresholds_fig5:

                subset = max_probs < t

                if subset.sum() == 0:
                    rows.append({
                        "threshold": t,
                        "sls_minus_acc": np.nan,
                        "gbdt_acc": np.nan,
                        "subset_ratio": 0
                    })
                    continue

                sls_minus_acc = (sls_preds[subset] == y_val[subset]).mean()
                gbdt_acc = (gbdt_preds[subset] == y_val[subset]).mean()

                rows.append({
                    "threshold": t,
                    "sls_minus_acc": sls_minus_acc,
                    "gbdt_acc": gbdt_acc,
                    "subset_ratio": subset.mean()
                })

        df = pd.DataFrame(rows)
        self.results["fig5"] = df.groupby("threshold").mean()


    # -------------------------------------------------------
    # PLOTTING
    # -------------------------------------------------------

    def plot_fig4(self):

        r=self.results["fig4"]

        plt.figure(figsize=(8,5))

        for name in r:
            plt.plot(self.thresholds_fig4,r[name],marker="o",label=name)

        plt.xlabel("Threshold")
        plt.ylabel("Accuracy")
        plt.title("Fig.4 Feature Ablation")
        plt.legend()
        plt.grid()
        plt.savefig("fig4_feature_ablation.png", dpi=300)
        plt.show()


    def plot_fig5(self):

        df=self.results["fig5"]

        plt.figure(figsize=(8,5))

        plt.plot(df.index,df["sls_minus_acc"],marker="o",label="SLS-")
        plt.plot(df.index,df["gbdt_acc"],marker="o",label="GBDT")

        plt.xlabel("Threshold")
        plt.ylabel("Accuracy")
        plt.title("Fig.5 GBDT Assistance Analysis")
        plt.legend()
        plt.grid()
        plt.savefig("fig5_feature_ablation.png", dpi=300)
        plt.show()


    # -------------------------------------------------------
    # RUN
    # -------------------------------------------------------

    def run(self):

        self.compute_fig4()
        self.compute_fig5()

        self.plot_fig4()
        self.plot_fig5()


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------

def main():

    import argparse

    parser=argparse.ArgumentParser()

    parser.add_argument("--config",default="configs/default.yaml")
    parser.add_argument("--data-dir",required=True)

    args=parser.parse_args()

    study=PaperExactAblation(args.config)

    events=study.load_pheme_data(args.data_dir)

    X,y=study.extract_features(events)
    study.X = X
    study.y=y

    study.setup_cv(X,y)

    study.run()


if __name__=="__main__":
    main()