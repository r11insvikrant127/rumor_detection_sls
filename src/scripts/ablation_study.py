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

        self.thresholds_fig4 = np.arange(0.52,0.66,0.01)
        self.thresholds_fig5 = np.arange(0.55,0.81,0.01)

        self.results = {}

    # -------------------------------------------------------
    # DATA LOADING
    # -------------------------------------------------------

    def load_pheme_data(self, data_dir):

        events = []
        data_dir = Path(data_dir)

        for event_dir in data_dir.iterdir():

            if not event_dir.is_dir():
                continue

            for label_dir in ["rumours","non-rumours"]:

                label_path = event_dir / label_dir

                if not label_path.exists():
                    continue

                for thread_dir in label_path.iterdir():

                    try:

                        event = {"tweets":[]}
                        event["label"] = 1 if label_dir=="rumours" else 0

                        # source tweet
                        source_dir = thread_dir / "source-tweet"

                        for f in source_dir.glob("*.json"):
                            with open(f) as fp:
                                s = json.load(fp)

                            event["tweets"].append({
                                "id":s.get("id_str",""),
                                "text":s.get("text",""),
                                "user":s.get("user",{}),
                                "created_at":s.get("created_at",""),
                                "response_to":None
                            })

                        # reactions
                        reactions_dir = thread_dir / "reactions"

                        if reactions_dir.exists():

                            for f in reactions_dir.glob("*.json"):

                                with open(f) as fp:
                                    r = json.load(fp)

                                event["tweets"].append({
                                    "id":r.get("id_str",""),
                                    "text":r.get("text",""),
                                    "user":r.get("user",{}),
                                    "created_at":r.get("created_at",""),
                                    "response_to":r.get("in_reply_to_status_id_str")
                                })

                        events.append(event)

                    except:
                        continue

        print("Loaded events:",len(events))
        return events


    # -------------------------------------------------------
    # FEATURE EXTRACTION
    # -------------------------------------------------------

    def extract_features(self,events):

        X=[]
        y=[]

        for e in events:

            try:

                feats=self.feature_extractor.extract_features(e)

                vec=[feats[name] for name in self.feature_names]

                X.append(vec)
                y.append(e["label"])

            except:
                pass

        X=np.array(X,dtype=np.float32)
        y=np.array(y)

        X=np.nan_to_num(X)

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

        trainer.train(train_loader,val_loader,epochs=100)

        model.eval()

        with torch.no_grad():

            X_tensor=torch.FloatTensor(X_val).unsqueeze(1).to(self.device)

            outputs=model(X_tensor)

            probs=torch.softmax(outputs,dim=1).cpu().numpy()

        return probs


    # -------------------------------------------------------
    # FIGURE 4
    # -------------------------------------------------------

    def compute_fig4(self):

        print("\nRunning Fig4 reproduction")

        models={
            "pSLS":self.feature_groups["propagation"],
            "uSLS":self.feature_groups["user"],
            "cSLS":self.feature_groups["content"],
            "SLS":list(range(31))
        }

        results={}

        for name,idx in models.items():

            fold_probs=[]
            fold_labels=[]

            for train_idx,val_idx in self.splits:

                X_train=self.X[train_idx][:,idx]
                y_train=self.y[train_idx]

                X_val=self.X[val_idx][:,idx]
                y_val=self.y[val_idx]

                probs=self.train_sls(
                    X_train,y_train,
                    X_val,y_val,
                    len(idx)
                )

                fold_probs.append(probs)
                fold_labels.append(y_val)

            threshold_acc=[]

            for t in self.thresholds_fig4:

                fold_acc=[]

                for probs,y_val in zip(fold_probs,fold_labels):

                    max_probs=np.max(probs,axis=1)
                    preds=np.argmax(probs,axis=1)

                    confident=max_probs>=t

                    if confident.sum()==0:
                        continue

                    acc=(preds[confident]==y_val[confident]).mean()

                    fold_acc.append(acc)

                if len(fold_acc)>0:
                    threshold_acc.append(np.mean(fold_acc))
                else:
                    threshold_acc.append(0)

            results[name]=threshold_acc

        self.results["fig4"]=results


    # -------------------------------------------------------
    # FIGURE 5
    # -------------------------------------------------------

    def compute_fig5(self):

        print("\nRunning Fig5 reproduction")

        rows=[]

        for train_idx,val_idx in self.splits:

            X_train=self.X[train_idx]
            y_train=self.y[train_idx]

            X_val=self.X[val_idx]
            y_val=self.y[val_idx]

            probs=self.train_sls(X_train,y_train,X_val,y_val,31)

            sls_preds=np.argmax(probs,axis=1)
            max_probs=np.max(probs,axis=1)

            gbdt=GBDTWrapper(**self.config.gbdt.__dict__)
            gbdt.fit(X_train,y_train)

            gbdt_preds=gbdt.predict(X_val)

            for t in self.thresholds_fig5:

                subset=max_probs<t

                if subset.sum()==0:
                    continue

                sls_minus_acc=(sls_preds[subset]==y_val[subset]).mean()
                gbdt_acc=(gbdt_preds[subset]==y_val[subset]).mean()

                rows.append({
                    "threshold":t,
                    "sls_minus_acc":sls_minus_acc,
                    "gbdt_acc":gbdt_acc,
                    "subset_ratio":subset.mean()
                })

        df=pd.DataFrame(rows)

        self.results["fig5"]=df.groupby("threshold").mean()


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

    normalizer=FeatureNormalizer()

    X=normalizer.fit_transform(X,study.feature_names)

    study.X=X
    study.y=y

    study.setup_cv(X,y)

    study.run()


if __name__=="__main__":
    main()