"""
Rumor Detection SLS Package
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Rumor Detection Team"

# Export main classes for easy import
from .preprocessing.feature_extractor import FeatureExtractor
from .models.sls import PaperExactSLS
from .training.trainer import SLSTrainer
from .utils.config import ConfigManager

__all__ = [
    "FeatureExtractor",
    "PaperExactSLS", 
    "SLSTrainer",
    "ConfigManager"
]