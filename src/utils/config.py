"""
Configuration management for rumor detection system.
Supports 56-feature system including depth-breadth weighting.
"""

import yaml
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    input_dim: int = 56  
    hidden_dim: int = 256
    lstm_hidden: int = 128
    dropout_rate: float = 0.15
    se_reduction: int = 16
    use_cosine_layer: bool = False  # For Circle Loss compatibility
    num_classes: int = 2  # rumor vs non-rumor
    
    # Feature selection (for ablation studies)
    use_propagation_features: bool = True
    use_user_features: bool = True
    use_content_features: bool = True
    use_depth_breadth_features: bool = True  # New 56-feature system
    use_optional_features: bool = True


@dataclass
class LossConfig:
    """Loss function configuration."""
    loss_type: str = "combined"  # circle, weighted_circle, combined, focal, ce
    margin: float = 0.25  # For Circle Loss
    gamma: float = 256  # For Circle Loss
    weight_circle: float = 0.7  # For Combined Loss
    alpha_focal: float = 0.25  # For Focal Loss
    gamma_focal: float = 2.0  # For Focal Loss
    label_smoothing: float = 0.1  # For regularization
    class_weights: List[float] = field(default_factory=lambda: [0.3, 0.7])  # For imbalanced data
    # 🔴 FIX 3: REMOVED normalize_logits (handled internally by loss functions)


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 10
    threshold: float = 0.5  # Updated from 0.57 for binary classification
    
    # Scheduler
    scheduler: str = "plateau"  # plateau, cosine, step
    scheduler_mode: str = "max"  # For plateau scheduler
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    
    # Early stopping - 🔴 FIX 4: This will be used by trainer
    early_stopping: bool = True
    early_stopping_patience: int = 10
    
    # Monitoring
    primary_metric: str = "f1"  # f1, accuracy, roc_auc
    logit_scale: float = 10.0  # For probability calibration


@dataclass
class DataConfig:
    """Data configuration."""
    feature_dim: int = 56  # Total features
    feature_groups: Dict[str, List[int]] = field(default_factory=dict)
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    normalize_features: bool = True
    
    # Dataset paths
    train_path: str = "data/train.csv"
    test_path: str = "data/test.csv"
    cache_dir: str = "data/cache"


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    use_wandb: bool = False
    wandb_project: str = "rumor-detection"
    wandb_entity: Optional[str] = None
    
    # Metrics
    compute_feature_importance: bool = True
    plot_confusion_matrix: bool = True
    plot_roc_curve: bool = True
    save_predictions: bool = True
    output_dir: str = "results"
    
    # Comparison
    compare_31vs56: bool = True  # Compare original vs new feature sets


@dataclass
class GBDTConfig:
    """GBDT fallback configuration."""
    enabled: bool = True
    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = 5
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    use_for_fallback: bool = True  # Use GBDT when model uncertain
    uncertainty_threshold: float = 0.3  # Max probability diff between classes


@dataclass
class ExperimentConfig:
    """Experiment tracking configuration."""
    experiment_name: str = "rumor_detection_56f"
    experiment_id: Optional[str] = None
    seed: int = 42
    num_runs: int = 5
    save_checkpoints: bool = True
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"


class ConfigManager:
    """
    Enhanced configuration manager with validation and feature tracking.
    """
    
    def __init__(self, config_path: Optional[str] = None, defaults: Optional[Dict] = None):
        # 🔴 FIX 1: Initialize logger
        self.logger = logging.getLogger(__name__)
        
        self.config = {}
        
        # Load from YAML if provided
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif defaults:
            self.config = defaults
        else:
            # Use default configuration
            self.config = self.get_default_config()
        
        # Initialize config objects
        self.model = ModelConfig(**self.config.get('model', {}))
        self.loss = LossConfig(**self.config.get('loss', {}))
        self.training = TrainingConfig(**self.config.get('training', {}))
        self.data = DataConfig(**self.config.get('data', {}))
        self.evaluation = EvaluationConfig(**self.config.get('evaluation', {}))
        self.gbdt = GBDTConfig(**self.config.get('gbdt', {}))
        self.experiment = ExperimentConfig(**self.config.get('experiment', {}))
        
        # Feature group tracking (56 features)
        self._setup_feature_groups()
        
        # Validate configuration
        self._validate_config()
    
    def _setup_feature_groups(self):
        """Setup feature groups for the 56-feature system."""
        # ✅ FIXED: Correct feature indices for 56 features
        # Indices follow 0-based indexing aligned with FeatureExtractor.FEATURE_NAMES (56 total)
        self.feature_groups = {
            # Propagation-based features (indices 0-5, 6 features)
            "propagation": list(range(0, 6)),
            
            # User-based features - Most influential (indices 6-11, 6 features)
            "user_source": list(range(6, 12)),
            
            # User-based features - Kernel aggregation (indices 12-19, 8 features)
            "user_kernel": list(range(12, 20)),
            
            # Content-based features - Most influential (indices 20-21, 2 features)
            "content_source": list(range(20, 22)),
            
            # Content-based features - Kernel aggregation (indices 22-33, 12 features)
            "content_kernel": list(range(22, 34)),
            
            # Tree metrics (indices 34-35, 2 features)
            "tree_metrics": list(range(34, 36)),
            
            # Response time metrics (indices 36-37, 2 features)
            "response_time": list(range(36, 38)),
            
            # Global content features (indices 38-40, 3 features)
            "global_content": list(range(38, 41)),
            
            # Optional features beyond paper (indices 41-44, 4 features)
            "optional": list(range(41, 45)),
            
            # Depth-breadth weighting features (indices 45-55, 11 features)
            "depth_breadth": list(range(45, 56))  
        }
        
        self.data.feature_groups = self.feature_groups
        
        # Log feature group summary
        self.logger.info("Feature groups configured for 56-feature system:")
        total_features = sum(len(indices) for indices in self.feature_groups.values())
        self.logger.info(f"Total features: {total_features}")
        
        for group, indices in self.feature_groups.items():
            self.logger.info(f"  {group}: {len(indices)} features (indices {indices[0]}-{indices[-1]})")
        
        # Verify total feature count
        if total_features != 56:
            self.logger.warning(f"Expected 56 features, but found {total_features}")
    
    def _validate_config(self):
        """Validate configuration consistency."""
        # Check feature dimensions
        if self.model.input_dim != self.data.feature_dim:
            # 🔴 FIX 1: Use self.logger instead of undefined logger
            self.logger.warning(
                f"Model input_dim ({self.model.input_dim}) doesn't match data feature_dim ({self.data.feature_dim})"
            )
        
        # Check loss configuration
        if self.loss.loss_type == "combined" and not (0 <= self.loss.weight_circle <= 1):
            raise ValueError(f"weight_circle must be between 0 and 1, got {self.loss.weight_circle}")
        
        if self.loss.margin < 0:
            raise ValueError(f"margin must be non-negative, got {self.loss.margin}")
        
        if self.loss.gamma <= 0:
            raise ValueError(f"gamma must be positive, got {self.loss.gamma}")
        
        # Check class weights
        if len(self.loss.class_weights) != self.model.num_classes:
            raise ValueError(
                f"class_weights length ({len(self.loss.class_weights)}) "
                f"doesn't match num_classes ({self.model.num_classes})"
            )
        
        # Check threshold
        if not (0 <= self.training.threshold <= 1):
            raise ValueError(f"threshold must be between 0 and 1, got {self.training.threshold}")
        
        # 🔴 FIX 4: Validate early stopping config
        if self.training.early_stopping and self.training.early_stopping_patience <= 0:
            raise ValueError(f"early_stopping_patience must be positive, got {self.training.early_stopping_patience}")
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get default configuration for 56-feature rumor detection."""
        return {
            "model": {
                "input_dim": 56,  # ✅ Already correct
                "hidden_dim": 256,
                "lstm_hidden": 128,
                "dropout_rate": 0.15,
                "se_reduction": 16,
                "use_cosine_layer": False,
                "num_classes": 2,
                "use_propagation_features": True,
                "use_user_features": True,
                "use_content_features": True,
                "use_depth_breadth_features": True,
                "use_optional_features": True
            },
            "loss": {
                "loss_type": "combined",
                "margin": 0.25,
                "gamma": 256,
                "weight_circle": 0.7,
                "alpha_focal": 0.25,
                "gamma_focal": 2.0,
                "label_smoothing": 0.1,
                "class_weights": [0.3, 0.7],
            },
            "training": {
                "batch_size": 32,
                "epochs": 100,
                "learning_rate": 0.001,
                "weight_decay": 1e-4,
                "grad_clip": 1.0,
                "patience": 10,
                "threshold": 0.5,
                "scheduler": "plateau",
                "scheduler_mode": "max",
                "scheduler_patience": 5,
                "scheduler_factor": 0.5,
                "early_stopping": True,
                "early_stopping_patience": 10,
                "primary_metric": "f1",
                "logit_scale": 1.0  # ✅ FIXED: Changed from 10.0 to 1.0 (no scaling)
            },
            "data": {
                "feature_dim": 56,  # ✅ Already correct
                "test_size": 0.2,
                "val_size": 0.1,
                "random_state": 42,
                "normalize_features": True,
                "train_path": "data/train.csv",
                "test_path": "data/test.csv",
                "cache_dir": "data/cache"
            },
            "evaluation": {
                "use_wandb": False,
                "wandb_project": "rumor-detection",
                "wandb_entity": None,
                "compute_feature_importance": True,
                "plot_confusion_matrix": True,
                "plot_roc_curve": True,
                "save_predictions": True,
                "output_dir": "results",
                "compare_31vs56": True  # ✅ Already correct
            },
            "gbdt": {
                "enabled": True,
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 5,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "use_for_fallback": True,
                "uncertainty_threshold": 0.3
            },
            "experiment": {
                "experiment_name": "rumor_detection_56f",
                "experiment_id": None,
                "seed": 42,
                "num_runs": 5,
                "save_checkpoints": True,
                "checkpoint_dir": "checkpoints",
                "log_dir": "logs"
            }
        }
    
    def get_loss_kwargs(self) -> Dict[str, Any]:
        """
        Get properly formatted loss kwargs for trainer.
        
        This handles the conditional passing of loss-specific parameters.
        """
        loss_kwargs = {
            'm': self.loss.margin,
            'gamma': self.loss.gamma
        }
        
        # Only add weight_circle for combined loss
        if self.loss.loss_type == 'combined':
            loss_kwargs['weight_circle'] = self.loss.weight_circle
        
        # 🔴 FIX 5: Add focal loss parameters when needed
        if self.loss.loss_type in ['focal', 'combined']:
            loss_kwargs['alpha_focal'] = self.loss.alpha_focal
            loss_kwargs['gamma_focal'] = self.loss.gamma_focal
        
        # Always add label smoothing (handled appropriately by loss functions)
        loss_kwargs['label_smoothing'] = self.loss.label_smoothing
        
        return loss_kwargs
    
    def get_feature_mask(self) -> List[bool]:
        """
        Get feature mask based on enabled feature groups.
        
        Returns:
            List of booleans indicating which features to use
        """
        total_features = self.data.feature_dim
        mask = [False] * total_features
        
        # Map config flags to feature groups - UPDATE TO MATCH TRAINER TAXONOMY
        group_enabled = {
            "propagation": self.model.use_propagation_features,
            "user_source": self.model.use_user_features,
            "user_kernel": self.model.use_user_features,
            "content_source": self.model.use_content_features,
            "content_kernel": self.model.use_content_features,
            "tree_metrics": self.model.use_propagation_features,  # ✅ FIXED: Part of propagation
            "response_time": self.model.use_propagation_features,  # ✅ FIXED: Part of propagation
            "global_content": True,  # Now in 'other' category
            "optional": self.model.use_optional_features,  # Now in 'other' category
            "depth_breadth": self.model.use_depth_breadth_features
        }
        
        # Apply masks
        for group, enabled in group_enabled.items():
            if group in self.feature_groups and enabled:
                for idx in self.feature_groups[group]:
                    if idx < total_features:
                        mask[idx] = True
        
        # Count enabled features
        enabled_count = sum(mask)
        disabled_count = total_features - enabled_count
        
        self.logger.info(f"Feature masking: {enabled_count} enabled, {disabled_count} disabled")
        
        return mask
    
    def get(self, key: str, default=None):
        """Get config value using dot notation."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        
        return value
    
    def update(self, updates: Dict[str, Any]):
        """Update configuration using dot notation."""
        for key, value in updates.items():
            keys = key.split('.')
            d = self.config
            
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            
            d[keys[-1]] = value
        
        # Re-initialize config objects
        self.__init__(defaults=self.config)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model": self.model.__dict__,
            "loss": self.loss.__dict__,
            "training": self.training.__dict__,
            "data": self.data.__dict__,
            "evaluation": self.evaluation.__dict__,
            "gbdt": self.gbdt.__dict__,
            "experiment": self.experiment.__dict__
        }
    
    def save(self, path: str):
        """Save configuration to YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
        
        self.logger.info(f"Configuration saved to: {path}")
    
    def save_json(self, path: str):
        """Save configuration to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        self.logger.info(f"Configuration saved to: {path}")
    
    def print_summary(self):
        """Print configuration summary."""
        print("=" * 60)
        print("CONFIGURATION SUMMARY (56 FEATURES)")
        print("=" * 60)
        
        print(f"\n📊 Model:")
        print(f"  Input dim: {self.model.input_dim} (56 features)")
        print(f"  Num classes: {self.model.num_classes}")
        print(f"  Use cosine layer: {self.model.use_cosine_layer}")
        
        print(f"\n🎯 Loss:")
        print(f"  Type: {self.loss.loss_type}")
        print(f"  Margin (m): {self.loss.margin}")
        print(f"  Class weights: {self.loss.class_weights}")
        if self.loss.loss_type in ['focal', 'combined']:
            print(f"  Focal alpha: {self.loss.alpha_focal}")
            print(f"  Focal gamma: {self.loss.gamma_focal}")
        
        print(f"\n🚀 Training:")
        print(f"  Batch size: {self.training.batch_size}")
        print(f"  Learning rate: {self.training.learning_rate}")
        print(f"  Early stopping: {self.training.early_stopping} (patience: {self.training.early_stopping_patience})")
        print(f"  Threshold: {self.training.threshold}")
        print(f"  Logit scale: {self.training.logit_scale} (1.0 = no scaling)")
        
        print(f"\n📈 Feature Groups (56 total):")
        total_count = 0
        for group, indices in self.feature_groups.items():
            enabled = True  # Simplified
            status = "✓" if enabled else "✗"
            count = len(indices)
            total_count += count
            print(f"  {status} {group}: {count:2d} features (indices {indices[0]:2d}-{indices[-1]:2d})")
        
        print(f"\n  Total features: {total_count}")
        print(f"\n🔧 GBDT Fallback: {'✓ Enabled' if self.gbdt.enabled else '✗ Disabled'}")
        print("=" * 60)

# Quick configuration helper
def load_config(config_path: Optional[str] = None) -> ConfigManager:
    """Helper function to load configuration."""
    return ConfigManager(config_path)


def create_ablation_configs(base_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Create ablation study configurations.
    
    Args:
        base_config: Base configuration
    
    Returns:
        List of configurations for ablation studies
    """
    configs = []
    
    # Base config
    configs.append(base_config.copy())
    
    # Ablation: Without depth-breadth features
    no_depth_breadth = base_config.copy()
    no_depth_breadth['model']['use_depth_breadth_features'] = False
    no_depth_breadth['experiment']['experiment_name'] = "no_depth_breadth"
    configs.append(no_depth_breadth)
    
    # Ablation: Without optional features
    no_optional = base_config.copy()
    no_optional['model']['use_optional_features'] = False
    no_optional['experiment']['experiment_name'] = "no_optional"
    configs.append(no_optional)
    
    # Ablation: Only propagation features
    only_propagation = base_config.copy()
    only_propagation['model'].update({
        'use_user_features': False,
        'use_content_features': False,
        'use_depth_breadth_features': False,
        'use_optional_features': False
    })
    only_propagation['experiment']['experiment_name'] = "only_propagation"
    configs.append(only_propagation)
    
    return configs