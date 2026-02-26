"""
Command-line scripts for training, prediction, and analysis.
"""
from .train import RumorDetectionTrainer
from .predict import RumorPredictor
from .ablation_study import PaperExactAblation

__all__ = [
    "RumorDetectionTrainer",
    "RumorPredictor", 
    "PaperExactAblation"
]