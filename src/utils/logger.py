"""
    Why researchers use this

The paper reports:

Accuracy

Precision

Recall

F1

5-fold CV results 


This logger helps you store those experiment results cleanly.
    
"""

"""
Paper-faithful logging utilities for SLS rumor detection.

Designed for reproducible experiments following:
Wei et al., 2021 (IJCNN)
"""

import logging
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any


# ============================================================
# PAPER LOGGER
# ============================================================

class PaperLogger:
    """
    Minimal experiment logger aligned with paper reproduction.
    """

    def __init__(
        self,
        name: str = "SLS",
        log_dir: str = "logs",
        level: int = logging.INFO,
    ):

        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{name}_{timestamp}.log"

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )

        # Console output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File output
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self.logger.info("Logger initialized")

    # --------------------------------------------------------

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def debug(self, msg: str):
        self.logger.debug(msg)

    # --------------------------------------------------------
    # METRIC LOGGING
    # --------------------------------------------------------

    def log_metrics(
        self,
        epoch: int,
        metrics: Dict[str, float]
    ):
        """
        Log training metrics per epoch.
        """

        metric_str = " | ".join(
            [f"{k}: {v:.4f}" for k, v in metrics.items()]
        )

        self.logger.info(f"Epoch {epoch} → {metric_str}")

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    def save_results(
        self,
        results: Dict[str, Any],
        filename: str = "results.json"
    ):

        path = self.log_dir / filename

        with open(path, "w") as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"Results saved → {path}")

    # --------------------------------------------------------

    def get_log_file(self) -> Path:
        return self.log_file


# ============================================================
# HELPER FUNCTION
# ============================================================

def setup_logger(name: str = "SLS") -> PaperLogger:
    """
    Create paper-faithful logger.
    """
    return PaperLogger(name=name)


# ============================================================
# SIMPLE LOGGER (BACKWARD COMPAT)
# ============================================================

def get_simple_logger(name: str = "SLS"):
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


if __name__ == "__main__":
    logger = setup_logger()
    logger.info("Paper logger ready.")