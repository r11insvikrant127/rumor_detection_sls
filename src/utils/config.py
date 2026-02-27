"""
What this file is used for

This file is NOT training code and NOT the model.

It is a Configuration Manager.

Think of it as the brain that tells every other file how to behave.

What it controls in your system

When you run:

train.py
predict.py
main.py

they do something like:

config = ConfigManager("config.yaml")

Then they read:

config.model.input_dim
config.loss.loss_type
config.training.batch_size
config.gbdt.threshold

So this file defines:

Component	Controlled by this file
Model architecture	input size, dropout, classes
Loss function	circle loss params
Training	LR, epochs, scheduler
Feature usage	which features enabled
Evaluation	metrics & outputs
GBDT fallback	uncertainty rule
Experiments	seeds & runs

👉 In short:

This file makes your whole pipeline paper-consistent.

"""

"""
Configuration management for PAPER-FAITHFUL SLS rumor detection system.

Implements configuration described in:
Wei et al., 2021 (IJCNN)
"A Novel and High-Accuracy Rumor Detection Approach using
Kernel Subtree and Deep Learning Networks"
"""

import yaml
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path


# ============================================================
# MODEL CONFIG
# ============================================================

@dataclass
class ModelConfig:
    """SLS architecture configuration (paper)."""

    input_dim: int = 31        # TABLE I features
    hidden_dim: int = 256
    lstm_hidden: int = 128
    dropout_rate: float = 0.15
    se_reduction: int = 16
    num_classes: int = 2

    # Paper feature groups
    use_propagation_features: bool = True
    use_user_features: bool = True
    use_content_features: bool = True


# ============================================================
# LOSS CONFIG
# ============================================================

@dataclass
class LossConfig:
    """Circle Loss configuration (paper)."""

    loss_type: str = "circle"
    margin: float = 0.25
    gamma: float = 256


# ============================================================
# TRAINING CONFIG
# ============================================================

@dataclass
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # Paper hybrid decision threshold
    threshold: float = 0.57

    scheduler: str = "plateau"
    scheduler_mode: str = "max"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5

    early_stopping: bool = True
    early_stopping_patience: int = 10

    primary_metric: str = "f1"


# ============================================================
# DATA CONFIG
# ============================================================

@dataclass
class DataConfig:
    feature_dim: int = 31
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    normalize_features: bool = True

    train_path: str = "data/train.csv"
    test_path: str = "data/test.csv"
    cache_dir: str = "data/cache"


# ============================================================
# EVALUATION CONFIG
# ============================================================

@dataclass
class EvaluationConfig:
    plot_confusion_matrix: bool = True
    plot_roc_curve: bool = True
    save_predictions: bool = True
    output_dir: str = "results"


# ============================================================
# GBDT CONFIG (Section IV-F)
# ============================================================

@dataclass
class GBDTConfig:
    enabled: bool = True
    threshold: float = 0.57

    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = 5
    subsample: float = 0.8
    colsample_bytree: float = 0.8

    use_for_fallback: bool = True


# ============================================================
# EXPERIMENT CONFIG
# ============================================================

@dataclass
class ExperimentConfig:
    experiment_name: str = "SLS_paper_faithful"
    experiment_id: Optional[str] = None
    seed: int = 42
    num_runs: int = 5     # 5-fold CV (paper)
    save_checkpoints: bool = True
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"


# ============================================================
# CONFIG MANAGER
# ============================================================

class ConfigManager:

    def __init__(self, config_path: Optional[str] = None):

        self.logger = logging.getLogger(__name__)

        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        else:
            config = {}

        self.model = ModelConfig(**config.get("model", {}))
        self.loss = LossConfig(**config.get("loss", {}))
        self.training = TrainingConfig(**config.get("training", {}))
        self.data = DataConfig(**config.get("data", {}))
        self.evaluation = EvaluationConfig(**config.get("evaluation", {}))
        self.gbdt = GBDTConfig(**config.get("gbdt", {}))
        self.experiment = ExperimentConfig(**config.get("experiment", {}))

        self._validate()

    # --------------------------------------------------------

    def _validate(self):

        if self.model.input_dim != self.data.feature_dim:
            raise ValueError(
                "Model input_dim must equal feature_dim (31 features)"
            )

        if not (0 <= self.training.threshold <= 1):
            raise ValueError("Threshold must be between 0 and 1")

    # --------------------------------------------------------

    def to_dict(self):

        return {
            "model": self.model.__dict__,
            "loss": self.loss.__dict__,
            "training": self.training.__dict__,
            "data": self.data.__dict__,
            "evaluation": self.evaluation.__dict__,
            "gbdt": self.gbdt.__dict__,
            "experiment": self.experiment.__dict__,
        }

    # --------------------------------------------------------

    def save(self, path: str):

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f)

        self.logger.info(f"Config saved → {path}")


# ============================================================
# HELPER
# ============================================================

def load_config(config_path: Optional[str] = None):
    return ConfigManager(config_path)