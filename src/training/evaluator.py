import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

class Evaluator:
    """Evaluation metrics for rumor detection as used in the paper."""
    
    @staticmethod
    def compute_metrics(y_true, y_pred, y_prob=None):
      
        metrics = {}
        
        # Basic classification metrics (all reported in paper)
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
        
        # ROC-AUC (reported in paper) - using hybrid probabilities
        if y_prob is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            except Exception as e:
                print(f"Warning: ROC-AUC calculation failed: {e}")
                metrics['roc_auc'] = 0.0
        
        # Confusion matrix (for reference)
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp'] = tn, fp, fn, tp
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics, title="Evaluation Results"):
        """Print metrics in a formatted way."""
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"{'='*60}")
        
        # Metrics reported in paper
        print("\n📊 Classification Metrics:")
        print("-" * 40)
        for key in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            if key in metrics:
                print(f"{key.upper():15s}: {metrics[key]:.4f}")
        
        # Confusion matrix
        if 'confusion_matrix' in metrics:
            print(f"\n📈 Confusion Matrix:")
            print("-" * 40)
            cm = metrics['confusion_matrix']
            print(f"True Negatives:  {metrics.get('tn', 0):4d}")
            print(f"False Positives: {metrics.get('fp', 0):4d}")
            print(f"False Negatives: {metrics.get('fn', 0):4d}")
            print(f"True Positives:  {metrics.get('tp', 0):4d}")
        
        # Uncertainty statistics if available
        if 'uncertain_ratio' in metrics:
            print(f"\n📊 Uncertainty Statistics:")
            print("-" * 40)
            print(f"Uncertain samples: {metrics.get('total_uncertain', 0):4d}")
            print(f"Uncertain ratio:   {metrics['uncertain_ratio']:.4f}")
    
    @staticmethod
    def compare_models(metrics_list, model_names=None):
        """
        Compare multiple models as in paper tables.
        
        Args:
            metrics_list: List of metric dictionaries
            model_names: List of model names
        """
        if model_names is None:
            model_names = [f"Model {i+1}" for i in range(len(metrics_list))]
        
        print(f"\n{'='*70}")
        print(f"MODEL COMPARISON")
        print(f"{'='*70}")
        
        # Table header
        print(f"\n{'Model':15s} | {'Accuracy':10s} | {'Precision':10s} | {'Recall':10s} | {'F1':10s} | {'AUC':10s}")
        print("-" * 75)
        
        for name, metrics in zip(model_names, metrics_list):
            acc = metrics.get('accuracy', 0)
            prec = metrics.get('precision', 0)
            rec = metrics.get('recall', 0)
            f1 = metrics.get('f1', 0)
            auc = metrics.get('roc_auc', 0)
            
            print(f"{name:15s} | {acc:10.4f} | {prec:10.4f} | {rec:10.4f} | {f1:10.4f} | {auc:10.4f}")


def evaluate_sklearn_model(model, X_test, y_test):
    """
    Evaluate sklearn-compatible models (GBDT, RF, etc.).
    
    Args:
        model: Trained sklearn model with predict and predict_proba
        X_test: Test features (numpy array)
        y_test: Test labels (numpy array)
        
    Returns:
        Dictionary of metrics
    """
    y_pred = model.predict(X_test)
    
    # Get probability predictions for ROC-AUC
    y_prob = None
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)[:, 1]  # Probability of class 1
        except:
            y_prob = model.predict_proba(X_test)
            if y_prob.shape[1] > 1:
                y_prob = y_prob[:, 1]
    
    # Compute metrics
    metrics = Evaluator.compute_metrics(y_test, y_pred, y_prob)
    
    return metrics


def evaluate_sls_model(model, test_loader, device='cuda'):
    """
    Evaluate PyTorch SLS model.
    
    Args:
        model: Trained SLS model
        test_loader: DataLoader with test data
        device: Device to run evaluation on
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                inputs = batch[0].to(device)
                labels = batch[1]
            else:
                inputs = batch.to(device)
                labels = None
                # If no labels provided, this won't work for evaluation
                raise ValueError("Test loader must return (inputs, labels)")
            
            # Forward pass
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)
            
            # Get predictions (class with highest probability)
            preds = torch.argmax(probs, dim=1)
            
            # Move to CPU and convert to numpy
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1
    
    # Convert to numpy arrays
    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    
    # Compute metrics
    metrics = Evaluator.compute_metrics(y_true, y_pred, y_prob)
    
    return metrics


def evaluate_sls_model_with_threshold(model, test_loader, threshold=0.57, gbdt_fallback=None, device='cuda'):
    """
    Evaluate SLS model with uncertainty threshold and optional GBDT fallback.
    All metrics (accuracy, F1, ROC-AUC) are computed using hybrid predictions/probabilities.
    
    Args:
        model: Trained SLS model
        test_loader: DataLoader with test data
        threshold: Uncertainty threshold (0.57 as per paper)
        gbdt_fallback: Optional GBDT model for uncertain samples
        device: Device to run evaluation on
        
    Returns:
        Dictionary of metrics with hybrid evaluation
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []  # Will store hybrid probabilities (SLS + GBDT)
    all_uncertain = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch[0].to(device), batch[1]
            batch_size = inputs.size(0)
            
            # Get SLS model predictions
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)
            
            # Initial predictions and probabilities
            max_probs, preds = probs.max(dim=1)
            uncertain_mask = max_probs < threshold
            
            # Convert to numpy for processing
            preds_np = preds.cpu().numpy()
            probs_np = probs[:, 1].cpu().numpy()  # Probability of class 1
            uncertain_np = uncertain_mask.cpu().numpy()
            
            # Apply GBDT fallback for uncertain samples - update both predictions AND probabilities
            if gbdt_fallback is not None and uncertain_np.any():
                uncertain_indices = np.where(uncertain_np)[0]
                uncertain_inputs = inputs[uncertain_indices].cpu().numpy()
                
                # Handle shape for sklearn
                if uncertain_inputs.ndim == 3:
                    uncertain_inputs = uncertain_inputs.squeeze(1)
                
                # Get GBDT predictions and probabilities
                gbdt_preds = gbdt_fallback.predict(uncertain_inputs)
                gbdt_probs = gbdt_fallback.predict_proba(uncertain_inputs)[:, 1]
                
                # Update hybrid predictions and probabilities
                preds_np[uncertain_indices] = gbdt_preds
                probs_np[uncertain_indices] = gbdt_probs
            
            # Store results
            all_preds.extend(preds_np)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs_np)  # Now storing hybrid probabilities
            all_uncertain.extend(uncertain_np)
    
    # Convert to numpy arrays
    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    
    # Compute metrics - ALL using hybrid predictions/probabilities
    metrics = Evaluator.compute_metrics(y_true, y_pred, y_prob)
    
    # Add uncertainty statistics
    metrics['uncertain_ratio'] = np.mean(all_uncertain)
    metrics['total_uncertain'] = np.sum(all_uncertain)
    
    return metrics


def evaluate_hybrid_system(sls_model, gbdt_model, test_loader, threshold=0.57, device='cuda'):
    """
    Comprehensive hybrid system evaluation with proper probability handling.
    
    Args:
        sls_model: Trained SLS model
        gbdt_model: Trained GBDT model for fallback
        test_loader: DataLoader with test data
        threshold: Uncertainty threshold
        device: Device for SLS model
        
    Returns:
        Dictionary with:
        - sls_only: Metrics using only SLS
        - gbdt_only: Metrics using only GBDT
        - hybrid: Metrics using hybrid system (with threshold)
        - uncertainty_stats: Statistics about uncertainty
    """
    results = {}
    
    # 1. Evaluate SLS only
    sls_metrics = evaluate_sls_model(sls_model, test_loader, device)
    results['sls_only'] = sls_metrics
    
    # 2. Extract data for GBDT evaluation
    all_inputs = []
    all_labels = []
    for batch in test_loader:
        inputs, labels = batch[0], batch[1]
        all_inputs.append(inputs.cpu().numpy())
        all_labels.append(labels.numpy())
    
    X_test = np.vstack(all_inputs)
    if X_test.ndim == 3:
        X_test = X_test.squeeze(1)
    y_test = np.concatenate(all_labels)
    
    # 3. Evaluate GBDT only
    gbdt_metrics = evaluate_sklearn_model(gbdt_model, X_test, y_test)
    results['gbdt_only'] = gbdt_metrics
    
    # 4. Evaluate hybrid system
    hybrid_metrics = evaluate_sls_model_with_threshold(
        sls_model, test_loader, threshold, gbdt_model, device
    )
    results['hybrid'] = hybrid_metrics
    
    return results


if __name__ == "__main__":
    # Example usage with mock data
    print("Testing Hybrid Evaluation with Proper Probability Handling")
    print("="*60)
    
    from sklearn.ensemble import GradientBoostingClassifier
    
    # Generate dummy data
    np.random.seed(42)
    n_samples = 100
    n_features = 31
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    # Train GBDT
    gbdt = GradientBoostingClassifier()
    gbdt.fit(X[:80], y[:80])
    
    # Mock SLS predictions
    y_pred_sls = np.random.randint(0, 2, 20)
    y_prob_sls = np.random.random(20)
    y_true = y[80:]
    
    # Mock hybrid with threshold
    threshold = 0.57
    uncertain = y_prob_sls < threshold
    
    # GBDT fallback
    y_pred_hybrid = y_pred_sls.copy()
    y_prob_hybrid = y_prob_sls.copy()
    
    if uncertain.any():
        # Mock GBDT predictions for uncertain samples
        gbdt_preds = np.random.randint(0, 2, uncertain.sum())
        gbdt_probs = np.random.random(uncertain.sum())
        y_pred_hybrid[uncertain] = gbdt_preds
        y_prob_hybrid[uncertain] = gbdt_probs
    
    # Compute metrics
    metrics_sls = Evaluator.compute_metrics(y_true, y_pred_sls, y_prob_sls)
    metrics_hybrid = Evaluator.compute_metrics(y_true, y_pred_hybrid, y_prob_hybrid)
    
    print("\n📊 SLS Only:")
    Evaluator.print_metrics(metrics_sls)
    
    print("\n📊 Hybrid (with GBDT fallback):")
    Evaluator.print_metrics(metrics_hybrid)
    
    print("\n✅ All metrics now use hybrid predictions/probabilities")
    print("   ✓ Accuracy: hybrid")
    print("   ✓ F1: hybrid")
    print("   ✓ ROC-AUC: hybrid")