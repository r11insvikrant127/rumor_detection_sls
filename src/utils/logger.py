"""
Enhanced logging with experiment tracking and structured logs.
"""

import logging
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import traceback


class ExperimentLogger:
    """
    Enhanced logger with experiment tracking capabilities.
    """
    
    def __init__(self, name: str = __name__, log_dir: str = "logs", 
                 experiment_id: Optional[str] = None, level: int = logging.INFO):
        self.name = name
        self.log_dir = Path(log_dir)
        self.experiment_id = experiment_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup loggers
        self.logger = self._setup_logger(name, level)
        self.metrics_logger = self._setup_metrics_logger()
        
        # Experiment metadata
        self.experiment_data = {
            "experiment_id": self.experiment_id,
            "start_time": datetime.now().isoformat(),
            "logs": []
        }
    
    def _setup_logger(self, name: str, level: int) -> logging.Logger:
        """Setup main logger."""
        logger = logging.getLogger(f"{name}.main")
        logger.setLevel(level)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Formatter with experiment ID
        formatter = logging.Formatter(
            f'%(asctime)s - %(name)s - EXP:{self.experiment_id} - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_file = self.log_dir / f"experiment_{self.experiment_id}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def _setup_metrics_logger(self) -> logging.Logger:
        """Setup separate logger for metrics."""
        metrics_logger = logging.getLogger(f"{self.name}.metrics")
        metrics_logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        metrics_logger.handlers.clear()
        
        # Metrics formatter (JSON for easy parsing)
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_record = {
                    "timestamp": datetime.now().isoformat(),
                    "experiment_id": self.experiment_id,
                    "level": record.levelname,
                    "message": record.getMessage(),
                    **getattr(record, "extra_data", {})
                }
                return json.dumps(log_record)
        
        # Metrics file handler
        metrics_file = self.log_dir / f"metrics_{self.experiment_id}.jsonl"
        metrics_handler = logging.FileHandler(metrics_file)
        metrics_handler.setFormatter(JSONFormatter())
        metrics_logger.addHandler(metrics_handler)
        
        return metrics_logger
    
    def info(self, message: str, extra_data: Optional[Dict] = None):
        """Log info message with optional extra data."""
        self.logger.info(message)
        if extra_data:
            self._log_structured("INFO", message, extra_data)
    
    def warning(self, message: str, extra_data: Optional[Dict] = None):
        """Log warning message."""
        self.logger.warning(message)
        if extra_data:
            self._log_structured("WARNING", message, extra_data)
    
    def error(self, message: str, exc_info: bool = True, extra_data: Optional[Dict] = None):
        """Log error message with exception info."""
        self.logger.error(message, exc_info=exc_info)
        if extra_data:
            self._log_structured("ERROR", message, extra_data)
    
    def critical(self, message: str, extra_data: Optional[Dict] = None):
        """Log critical message."""
        self.logger.critical(message)
        if extra_data:
            self._log_structured("CRITICAL", message, extra_data)
    
    def debug(self, message: str, extra_data: Optional[Dict] = None):
        """Log debug message."""
        self.logger.debug(message)
        if extra_data:
            self._log_structured("DEBUG", message, extra_data)
    
    def _log_structured(self, level: str, message: str, extra_data: Dict):
        """Log structured data to metrics logger."""
        log_record = {
            "level": level,
            "message": message,
            **extra_data
        }
        
        # Store in experiment data
        self.experiment_data["logs"].append({
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            "data": extra_data
        })
        
        # Log to metrics file
        self.metrics_logger.info(message, extra={"extra_data": log_record})
    
    def log_metric(self, metric_name: str, value: float, step: Optional[int] = None, 
                   epoch: Optional[int] = None, extra: Optional[Dict] = None):
        """Log a metric value."""
        metric_data = {
            "metric": metric_name,
            "value": value,
            "step": step,
            "epoch": epoch,
            "timestamp": datetime.now().isoformat()
        }
        
        if extra:
            metric_data.update(extra)
        
        self._log_structured("METRIC", f"Metric {metric_name}: {value}", metric_data)
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration."""
        config_file = self.log_dir / f"config_{self.experiment_id}.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.info(f"Configuration saved to {config_file}")
        self._log_structured("CONFIG", "Experiment configuration", {"config": config})
    
    def log_experiment_start(self, config: Dict[str, Any]):
        """Log experiment start."""
        self.info(f"Starting experiment: {self.experiment_id}")
        self.log_config(config)
        
        self.experiment_data["config"] = config
        self.experiment_data["status"] = "running"
    
    def log_experiment_end(self, success: bool = True, results: Optional[Dict] = None):
        """Log experiment end."""
        self.experiment_data["end_time"] = datetime.now().isoformat()
        self.experiment_data["status"] = "completed" if success else "failed"
        
        if results:
            self.experiment_data["results"] = results
            self.info(f"Experiment completed with results: {results}")
        else:
            self.info(f"Experiment {'completed' if success else 'failed'}")
        
        # Save experiment summary
        summary_file = self.log_dir / f"summary_{self.experiment_id}.json"
        with open(summary_file, 'w') as f:
            json.dump(self.experiment_data, f, indent=2)
    
    def log_exception(self, exception: Exception, context: Optional[str] = None):
        """Log exception with context."""
        exc_info = {
            "type": type(exception).__name__,
            "message": str(exception),
            "traceback": traceback.format_exc()
        }
        
        if context:
            exc_info["context"] = context
        
        self.error(f"Exception occurred: {exception}", extra_data={"exception": exc_info})
    
    def get_log_file(self) -> Path:
        """Get path to main log file."""
        return self.log_dir / f"experiment_{self.experiment_id}.log"
    
    def get_metrics_file(self) -> Path:
        """Get path to metrics file."""
        return self.log_dir / f"metrics_{self.experiment_id}.jsonl"


def setup_logger(name: str = __name__, log_file: Optional[str] = None, 
                 level: int = logging.INFO, experiment_id: Optional[str] = None) -> ExperimentLogger:
    """
    Setup enhanced logger with experiment tracking.
    
    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level
        experiment_id: Experiment identifier
    
    Returns:
        Enhanced logger instance
    """
    return ExperimentLogger(name=name, log_dir="logs", 
                           experiment_id=experiment_id, level=level)


# Simple logger for backward compatibility
class SimpleLogger:
    """Simple logger for quick use."""
    
    @staticmethod
    def get_logger(name: str = __name__, level: int = logging.INFO):
        """Get simple logger."""
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Clear existing handlers
        if not logger.handlers:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger