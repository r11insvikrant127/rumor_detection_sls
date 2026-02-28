"""
    This file = training & experiment utilities.

It does support operations, not modeling.

Think of your project layers:

Paper Model (SLS)
    ↑
trainer.py
    ↑
utils/helpers.py   ← THIS FILE
What it actually helps with

During training & testing, other files call functions here:

Function	Used for
normalize_features()	Z-score normalization
create_data_loader()	PyTorch batching
split_data()	train/val/test split
set_seed()	reproducibility
save_model()	checkpoint saving
load_model()	reload trained SLS
plot_training_history()	training curves
save_results()	experiment outputs

So this file provides infrastructure, not algorithm logic.

Where it appears in workflow
CSV features
     ↓
normalize_features()
     ↓
DataLoader
     ↓
SLS Training
     ↓
save_model()
     ↓
predict.py
     ↓
save_results()
    
"""

"""
Utility functions for PAPER-FAITHFUL SLS rumor detection system.

Aligned with:
Wei et al., 2021 (IJCNN)
"""

import numpy as np
import torch
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any
import json
from sklearn.preprocessing import StandardScaler
from datetime import datetime


# ============================================================
# FEATURE NORMALIZATION (Eq. 2 — Paper)
# ============================================================

def normalize_features(
    features: np.ndarray,
    scaler: Optional[StandardScaler] = None
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Z-score normalization used in the paper.
    """

    if scaler is None:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    else:
        features = scaler.transform(features)

    return features, scaler


# ============================================================
# DATALOADER (SLS INPUT FORMAT)
# ============================================================

from torch.utils.data import WeightedRandomSampler


def create_data_loader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    add_channel_dim: bool = True,
    use_weighted_sampler: bool = False,
) -> torch.utils.data.DataLoader:
    """
    Create DataLoader with required shape (N,1,L).

    If use_weighted_sampler=True:
        performs class-balanced sampling WITHOUT modifying loss.
    """

    x = torch.FloatTensor(features)

    if add_channel_dim:
        x = x.unsqueeze(1)

    y = torch.LongTensor(labels)

    dataset = torch.utils.data.TensorDataset(x, y)

    sampler = None

    # --------------------------------------------------
    # CLASS-BALANCED SAMPLING (paper-safe improvement)
    # --------------------------------------------------
    if use_weighted_sampler:

        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts

        sample_weights = class_weights[labels]

        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
        )

        shuffle = False  # sampler controls ordering

    # --------------------------------------------------
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
    )


# ============================================================
# DEVICE
# ============================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        print("Using GPU")
        return torch.device("cuda")
    print("Using CPU")
    return torch.device("cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = 42):

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# MODEL SAVE / LOAD
# ============================================================

def save_model(model: torch.nn.Module,
               scaler: StandardScaler,
               path: str,
               metadata: Optional[Dict] = None):

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "scaler": scaler,
        "metadata": metadata or {}
    }, path)

    print(f"Model saved → {path}")


def load_model(path: str,
               model_class,
               device: Optional[torch.device] = None):

    device = device or get_device()

    checkpoint = torch.load(path, map_location=device)

    model = model_class()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    scaler = checkpoint["scaler"]

    print(f"Model loaded ← {path}")

    return model, scaler, checkpoint.get("metadata", {})


# ============================================================
# RESULTS
# ============================================================

def save_results(results: Dict[str, Any], path: str):

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    if "predictions" in results:
        pd.DataFrame(results["predictions"]).to_csv(
            path.replace(".json", ".csv"),
            index=False
        )

    print(f"Results saved → {path}")


# ============================================================
# EXPERIMENT DIRECTORY
# ============================================================

def create_experiment_dir(name: str,
                          base_dir: str = "experiments") -> Path:

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(base_dir) / f"{name}_{timestamp}"

    for sub in ["models", "plots", "results", "logs"]:
        (path / sub).mkdir(parents=True, exist_ok=True)

    print(f"Experiment dir: {path}")
    return path


if __name__ == "__main__":
    print("Paper-faithful utils loaded.")