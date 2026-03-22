import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


class Evaluator:
    # =====================================================
    # CORE METRICS
    # =====================================================
    @staticmethod
    def compute_metrics(y_true, y_pred):
        assert len(y_true) == len(y_pred), "Mismatch in prediction and label length"

        metrics = {}

        # ---- Overall metrics (paper standard) ----
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(
            y_true, y_pred, average='binary', pos_label=1, zero_division=0
        )
        metrics['recall'] = recall_score(
            y_true, y_pred, average='binary', pos_label=1, zero_division=0
        )
        metrics['f1'] = f1_score(
            y_true, y_pred, average='binary', zero_division=0
        )
        metrics['f1_micro'] = f1_score(
            y_true, y_pred, average='micro', zero_division=0
        )

        # ---- Per-class metrics (matches Table III) ----
        precision_per_class = precision_score(
            y_true, y_pred, labels=[0, 1], average=None, zero_division=0
        )
        recall_per_class = recall_score(
            y_true, y_pred, labels=[0, 1], average=None, zero_division=0
        )
        f1_per_class = f1_score(
            y_true, y_pred, labels=[0, 1], average=None, zero_division=0
        )

        # Assuming: 0 = non-rumor, 1 = rumor
        metrics['precision_non_rumor'] = precision_per_class[0]
        metrics['precision_rumor'] = precision_per_class[1]

        metrics['recall_non_rumor'] = recall_per_class[0]
        metrics['recall_rumor'] = recall_per_class[1]

        metrics['f1_non_rumor'] = f1_per_class[0]
        metrics['f1_rumor'] = f1_per_class[1]

        # ---- Macro F1 (useful for research) ----
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')

        # ---- Confusion matrix ----
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp'] = tn, fp, fn, tp

        # ---- Class balance (debug insight) ----
        metrics['rumor_ratio'] = np.mean(y_true)

        return metrics

    # =====================================================
    # PRINT METRICS
    # =====================================================
    @staticmethod
    def print_metrics(metrics, title="Evaluation Results"):
        print(f"\n{'='*60}")
        print(title)
        print(f"{'='*60}")

        # ---- Overall ----
        print("\n📊 Overall Metrics:")
        print("-" * 40)
        for key in ['accuracy', 'precision', 'recall', 'f1', 'f1_micro', 'f1_macro']:
            if key in metrics:
                print(f"{key.upper():15s}: {metrics[key]:.4f}")

        # ---- Per-class ----
        print("\n📊 Per-Class Metrics:")
        print("-" * 40)
        print(f"{'Class':12s} | {'Precision':10s} | {'Recall':10s} | {'F1':10s}")
        print("-" * 50)
        print(f"{'Non-Rumor':12s} | "
              f"{metrics.get('precision_non_rumor', 0):10.4f} | "
              f"{metrics.get('recall_non_rumor', 0):10.4f} | "
              f"{metrics.get('f1_non_rumor', 0):10.4f}")

        print(f"{'Rumor':12s} | "
              f"{metrics.get('precision_rumor', 0):10.4f} | "
              f"{metrics.get('recall_rumor', 0):10.4f} | "
              f"{metrics.get('f1_rumor', 0):10.4f}")

        # ---- Confusion Matrix ----
        if 'confusion_matrix' in metrics:
            print(f"\n📈 Confusion Matrix:")
            print("-" * 40)
            print(f"True Negatives:  {metrics.get('tn', 0):4d}")
            print(f"False Positives: {metrics.get('fp', 0):4d}")
            print(f"False Negatives: {metrics.get('fn', 0):4d}")
            print(f"True Positives:  {metrics.get('tp', 0):4d}")

        # ---- Debug info ----
        print(f"\n📌 Rumor Ratio: {metrics.get('rumor_ratio', 0):.4f}")

    # =====================================================
    # MODEL COMPARISON (TABLE STYLE)
    # =====================================================
    @staticmethod
    def compare_models(metrics_list, model_names=None):

        if model_names is None:
            model_names = [f"Model {i+1}" for i in range(len(metrics_list))]

        print(f"\n{'='*80}")
        print("MODEL COMPARISON")
        print(f"{'='*80}")

        print(f"\n{'Model':15s} | {'Accuracy':10s} | {'Precision':10s} | {'Recall':10s} | {'F1':10s}")
        print("-" * 75)

        for name, metrics in zip(model_names, metrics_list):
            print(f"{name:15s} | "
                  f"{metrics.get('accuracy', 0):10.4f} | "
                  f"{metrics.get('precision', 0):10.4f} | "
                  f"{metrics.get('recall', 0):10.4f} | "
                  f"{metrics.get('f1', 0):10.4f}")


# =====================================================
# SKLEARN MODEL EVALUATION (GBDT etc.)
# =====================================================
def evaluate_sklearn_model(model, X_test, y_test):
    """
    Evaluate sklearn models (GBDT etc.)
    """
    y_pred = model.predict(X_test)
    return Evaluator.compute_metrics(y_test, y_pred)


# =====================================================
# PYTORCH MODEL EVALUATION (SLS only)
# =====================================================
def evaluate_sls_model(model, test_loader, device='cuda'):
    """
    Evaluate SLS model WITHOUT threshold (pure SLS)
    """
    model.eval()

    all_preds = []
    all_labels = []

    import torch

    with torch.no_grad():
        for inputs, labels in test_loader:

            inputs = inputs.to(device)

            similarities = model(inputs)
            preds = torch.argmax(similarities, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return Evaluator.compute_metrics(
        np.array(all_labels),
        np.array(all_preds)
    )


# =====================================================
# SIMPLE TEST
# =====================================================
if __name__ == "__main__":
    print("Testing Evaluator (Paper-Faithful)")
    print("=" * 60)

    np.random.seed(42)

    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)

    metrics = Evaluator.compute_metrics(y_true, y_pred)
    Evaluator.print_metrics(metrics)