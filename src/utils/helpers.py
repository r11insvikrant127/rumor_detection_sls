"""
Utility functions for rumor detection system.
"""

import numpy as np
import torch
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, List, Any
import json
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime



def normalize_features(features: np.ndarray, scaler: Optional[object] = None,
                       method: str = 'standard') -> Tuple[np.ndarray, Optional[object]]:
    """
    Normalize features using specified method.
    
    Args:
        features: Input features
        scaler: Pre-fitted scaler (if None, fit new one)
        method: 'standard' (Z-score) or 'robust' (robust to outliers)
    
    Returns:
        normalized_features, scaler
    """
    if method == 'standard':
        ScalerClass = StandardScaler
    elif method == 'robust':
        ScalerClass = RobustScaler
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    if scaler is None:
        scaler = ScalerClass()
        normalized = scaler.fit_transform(features)
        return normalized, scaler
    else:
        normalized = scaler.transform(features)
        return normalized, scaler


def create_data_loader(features: np.ndarray, labels: np.ndarray, 
                       batch_size: int = 32, shuffle: bool = True,
                       add_channel_dim: bool = False) -> torch.utils.data.DataLoader:
    """
    Create PyTorch DataLoader.
    
    Args:
        features: Input features
        labels: Target labels
        batch_size: Batch size
        shuffle: Whether to shuffle data
        add_channel_dim: Add channel dimension for CNN compatibility
    
    Returns:
        DataLoader
    """
    features_tensor = torch.FloatTensor(features)
    
    if add_channel_dim:
        features_tensor = features_tensor.unsqueeze(1)  # Add channel dimension
    
    labels_tensor = torch.LongTensor(labels)
    
    dataset = torch.utils.data.TensorDataset(features_tensor, labels_tensor)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=0, pin_memory=True  # Adjust num_workers based on your system
    )
    
    return loader


def split_data(features: np.ndarray, labels: np.ndarray, 
               test_size: float = 0.2, val_size: float = 0.1,
               random_state: int = 42, stratify: bool = True) -> Tuple:
    """
    Split data into train/validation/test sets.
    
    Args:
        features: Input features
        labels: Target labels
        test_size: Test set proportion
        val_size: Validation set proportion (of training data)
        random_state: Random seed
        stratify: Whether to stratify based on labels
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First split: train+val vs test
    stratify_labels = labels if stratify else None
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        features, labels, test_size=test_size, 
        random_state=random_state, stratify=stratify_labels
    )
    
    # Second split: train vs val
    if stratify:
        # Recalculate stratify for train/val split
        _, val_counts = np.unique(y_train_val, return_counts=True)
        if val_counts.min() < 2:  # Not enough samples for stratification
            stratify_labels = None
        else:
            stratify_labels = y_train_val
    
    val_size_adjusted = val_size / (1 - test_size)  # Adjust for initial split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size_adjusted,
        random_state=random_state, stratify=stratify_labels
    )
    
    print(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    print(f"Class distribution - Train: {np.bincount(y_train)}, "
          f"Val: {np.bincount(y_val)}, Test: {np.bincount(y_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Get available device.
    
    Args:
        prefer_gpu: Whether to prefer GPU if available
    
    Returns:
        torch.device
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to: {seed}")


def save_model(model: torch.nn.Module, path: str, metadata: Optional[Dict] = None):
    """
    Save model with metadata.
    
    Args:
        model: PyTorch model
        path: Save path
        metadata: Additional metadata
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'metadata': metadata or {}
    }
    
    torch.save(save_dict, path)
    print(f"Model saved to: {path}")


def load_model(path: str, model_class: Optional[torch.nn.Module] = None,
               device: Optional[torch.device] = None) -> Tuple[torch.nn.Module, Dict]:
    """
    Load model with metadata.
    
    Args:
        path: Model path
        model_class: Model class (if None, metadata must contain class info)
        device: Device to load to
    
    Returns:
        model, metadata
    """
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(path, map_location=device)
    
    if model_class is None:
        # Try to get model class from metadata
        model_class_name = checkpoint.get('model_class')
        if not model_class_name:
            raise ValueError("Model class not specified and not in checkpoint")
        
        # You'll need to import your model class here
        # This is a placeholder - you should implement based on your project structure
        raise ValueError(f"Please provide model_class for {model_class_name}")
    
    model = model_class()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from: {path}")
    return model, checkpoint.get('metadata', {})


def calculate_class_weights(labels: np.ndarray, method: str = 'balanced') -> np.ndarray:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        labels: Array of labels
        method: 'balanced' (inverse frequency) or 'uniform'
    
    Returns:
        Array of class weights
    """
    unique_classes, class_counts = np.unique(labels, return_counts=True)
    n_classes = len(unique_classes)
    
    if method == 'balanced':
        weights = class_counts.sum() / (n_classes * class_counts)
    elif method == 'uniform':
        weights = np.ones(n_classes)
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    # Normalize weights
    weights = weights / weights.sum()
    
    print(f"Class counts: {dict(zip(unique_classes, class_counts))}")
    print(f"Class weights: {dict(zip(unique_classes, weights))}")
    
    return weights


def plot_training_history(train_losses: List[float], val_metrics: List[Dict], 
                          save_path: Optional[str] = None):
    """
    Plot training history.
    
    Args:
        train_losses: List of training losses
        val_metrics: List of validation metrics dictionaries
        save_path: Path to save plot
    """
    epochs = range(1, len(train_losses) + 1)
    
    # Extract validation metrics
    val_losses = [m.get('val_loss', 0) for m in val_metrics]
    val_accuracies = [m.get('accuracy', 0) for m in val_metrics]
    val_f1s = [m.get('f1', 0) for m in val_metrics]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Loss plot
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[0, 1].plot(epochs, val_accuracies, 'g-', label='Val Accuracy', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    # F1 plot
    axes[1, 0].plot(epochs, val_f1s, 'purple-', label='Val F1', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Validation F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    # Learning rate plot (if available)
    if 'learning_rate' in val_metrics[0]:
        lrs = [m.get('learning_rate', 0) for m in val_metrics]
        axes[1, 1].plot(epochs, lrs, 'orange-', label='Learning Rate', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    
    plt.show()


def save_results(results: Dict[str, Any], path: str):
    """
    Save experiment results.
    
    Args:
        results: Results dictionary
        path: Save path
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Also save as CSV if results contain tabular data
    csv_path = path.replace('.json', '.csv')
    if 'predictions' in results:
        df = pd.DataFrame(results['predictions'])
        df.to_csv(csv_path, index=False)
    
    print(f"Results saved to: {path}")


def load_results(path: str) -> Dict[str, Any]:
    """
    Load experiment results.
    
    Args:
        path: Results file path
    
    Returns:
        Results dictionary
    """
    with open(path, 'r') as f:
        results = json.load(f)
    
    return results


def create_experiment_dir(experiment_name: str, base_dir: str = "experiments") -> Path:
    """
    Create experiment directory with timestamp.
    
    Args:
        experiment_name: Name of experiment
        base_dir: Base directory
    
    Returns:
        Path to experiment directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / "models").mkdir(exist_ok=True)
    (exp_dir / "plots").mkdir(exist_ok=True)
    (exp_dir / "results").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    
    print(f"Experiment directory created: {exp_dir}")
    return exp_dir


# Backward compatibility
if __name__ == "__main__":
    print("Utils module loaded successfully")