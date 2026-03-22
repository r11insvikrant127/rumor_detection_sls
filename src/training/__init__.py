"""
Training utilities and trainers for rumor detection models.
"""

# ---- Core SLS pipeline ----
from .trainer import SLSTrainer
from .loss import CircleLoss
from .evaluator import Evaluator

# ---- New model trainers ----
from .bigcn_trainer import BiGCNTrainer
from .rvnn_trainer import RvNNTrainer
from .ppc_trainer import PPCTrainer

__all__ = [
    # Core
    "SLSTrainer",
    "CircleLoss",
    "Evaluator",

    # Graph / Tree / Sequence models
    "BiGCNTrainer",
    "RvNNTrainer",
    "PPCTrainer",
]