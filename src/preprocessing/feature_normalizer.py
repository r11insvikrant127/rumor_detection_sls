import numpy as np


class FeatureNormalizer:
    """
    Strict implementation of paper normalization:

        f_hat(i,k) = (f(i,k) - mean_k) / std_k
    """

    def __init__(self):
        self.mean = None
        self.std = None
        self.is_fitted = False
        self.feature_names = None  # optional metadata

    # --------------------------------------------------
    # FIT
    # --------------------------------------------------
    def fit(self, X: np.ndarray, feature_names=None):
        """
        Compute feature-wise mean and std.

        Args:
            X: shape (N_events, N_features)
        """
        assert X.ndim == 2

        self.feature_names = feature_names

        # mean_k
        self.mean = np.mean(X, axis=0)

        # sigma(f_k)
        self.std = np.std(X, axis=0)

        # avoid divide-by-zero
        self.std[self.std == 0] = 1.0

        self.is_fitted = True
        return self

    # --------------------------------------------------
    # TRANSFORM
    # --------------------------------------------------
    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted first.")

        return (X - self.mean) / self.std

    # --------------------------------------------------
    # FIT + TRANSFORM
    # --------------------------------------------------
    def fit_transform(self, X: np.ndarray, feature_names=None) -> np.ndarray:
        self.fit(X, feature_names)
        return self.transform(X)