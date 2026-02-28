from __future__ import annotations

"""
Configuration Manager for SLS Rumor Detection System.

This file defines ALL experiment configuration used by:

    train.py
    predict.py
    main.py

Usage:
    config = ConfigManager("configs/default.yaml")

Then access:
    config.model.input_dim
    config.training.batch_size
    config.gbdt.threshold
"""

import yaml
import logging
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime


# ============================================================
# MODEL CONFIG
# ============================================================

@dataclass
class ModelConfig:
    input_dim: int = 31
    hidden_dim: int = 256
    lstm_hidden: int = 128
    dropout_rate: float = 0.15
    se_reduction: int = 16
    num_classes: int = 2

    use_propagation_features: bool = True
    use_user_features: bool = True
    use_content_features: bool = True


# ============================================================
# LOSS CONFIG
# ============================================================

@dataclass
class LossConfig:
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

    # ✅ inference calibration (does NOT change training objective)
    temperature: float = 1.0

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
# GBDT CONFIG (Hybrid fallback — Paper Section IV-F)
# ============================================================

@dataclass
class GBDTConfig:
    enabled: bool = True

    # ✅ SINGLE SOURCE OF TRUTH FOR THRESHOLD
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
    num_runs: int = 5
    save_checkpoints: bool = True
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"


# ============================================================
# CONFIG MANAGER
# ============================================================

class ConfigManager:

    def __init__(self, config_path: Optional[str] = None):

        self.logger = logging.getLogger("ConfigManager")

        config = self._load_yaml(config_path)

        self.model = ModelConfig(**config.get("model", {}))
        self.loss = LossConfig(**config.get("loss", {}))
        self.training = TrainingConfig(**config.get("training", {}))
        self.data = DataConfig(**config.get("data", {}))
        self.evaluation = EvaluationConfig(**config.get("evaluation", {}))
        self.gbdt = GBDTConfig(**config.get("gbdt", {}))
        self.experiment = ExperimentConfig(**config.get("experiment", {}))

        # auto experiment id
        if self.experiment.experiment_id is None:
            self.experiment.experiment_id = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

        self._validate()

    # --------------------------------------------------------

    def _load_yaml(self, path: Optional[str]) -> Dict[str, Any]:

        if path is None:
            return {}

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        return data

    # --------------------------------------------------------

    def _validate(self):

        # feature consistency
        if self.model.input_dim != self.data.feature_dim:
            raise ValueError(
                "Model input_dim must equal feature_dim (31 features)"
            )

        # threshold validation (correct location)
        if not (0 <= self.gbdt.threshold <= 1):
            raise ValueError("GBDT threshold must be in [0,1]")

        if self.loss.loss_type != "circle":
            raise ValueError("Paper requires Circle Loss")

        if self.training.batch_size <= 0:
            raise ValueError("Batch size must be positive")

    # --------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:

        return {
            "model": asdict(self.model),
            "loss": asdict(self.loss),
            "training": asdict(self.training),
            "data": asdict(self.data),
            "evaluation": asdict(self.evaluation),
            "gbdt": asdict(self.gbdt),
            "experiment": asdict(self.experiment),
        }

    # --------------------------------------------------------

    def save(self, path: str):

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

        self.logger.info(f"Config saved → {path}")


# ============================================================
# HELPER
# ============================================================

def load_config(config_path: Optional[str] = None) -> ConfigManager:
    return ConfigManager(config_path)