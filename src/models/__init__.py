"""
Model architectures for rumor detection.
"""
from .sls import PaperExactSLS
from .gbdt_wrapper import GBDTWrapper

__all__ = [
    "PaperExactSLS",
    "GBDTWrapper"
]