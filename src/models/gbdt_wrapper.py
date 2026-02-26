from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import joblib

class GBDTWrapper:
    """GBDT classifier for auxiliary predictions."""
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=5):
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )
        self.feature_count = None  # Optional: track feature count
        
    def fit(self, X_train, y_train):
        """Train GBDT model."""
        self.model.fit(X_train, y_train)
        self.feature_count = X_train.shape[1]  # Optional: store feature count
        print(f"GBDT trained on {self.feature_count} features")  # Debug info
        
    def predict(self, X):
        """Predict using GBDT."""
        if self.feature_count and X.shape[1] != self.feature_count:
            raise ValueError(
                f"GBDT feature mismatch: expected {self.feature_count}, got {X.shape[1]}"
            )
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities."""
        return self.model.predict_proba(X)
    
    def save(self, path):
        """Save model to disk."""
        save_data = {
            'model': self.model,
            'feature_count': self.feature_count
        }
        joblib.dump(save_data, path)
        
    def load(self, path):
        """Load model from disk."""
        loaded = joblib.load(path)
        self.model = loaded['model']
        self.feature_count = loaded.get('feature_count')
   
    # Optional: Add feature importance analysis
    def get_feature_importance(self, feature_names=None):
        """Get feature importance scores."""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            if feature_names and len(feature_names) == len(importances):
                # Return sorted feature importances
                indices = np.argsort(importances)[::-1]
                return [(feature_names[i], importances[i]) for i in indices]
            return importances
        return None
    
    def get_uncertain_indices(self, sls_probs, threshold=0.57):

        sls_probs = np.asarray(sls_probs)
        if sls_probs.ndim == 2:
            max_probs = np.max(sls_probs, axis=1)
        else:
            max_probs = sls_probs

        return np.where(max_probs < threshold)[0]
