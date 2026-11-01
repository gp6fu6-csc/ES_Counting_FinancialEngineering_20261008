"""Discrete Residual Classifier for Direct Sheet Count Integer Optimization.

Maps continuous residual delta = GT - period_count into discrete integer offsets,
training a multi-class classifier (XGBClassifier / LGBMClassifier) to predict
the exact integer correction directly.
"""
import numpy as np
from sklearn.preprocessing import LabelEncoder


class DiscreteResidualClassifier:
    """Discrete residual classifier that directly outputs integer sheet corrections."""

    def __init__(self, base_model=None, kind="xgb", max_offset=4, device="cpu", params=None):
        """
        Args:
            base_model: Pre-instantiated classifier (optional).
            kind: 'xgb' or 'lgbm'.
            max_offset: Maximum integer offset clip range [-max_offset, +max_offset] (default: 4).
            device: 'cuda' or 'cpu'.
            params: Dictionary of hyperparameters.
        """
        self.kind = kind
        self.max_offset = int(max_offset)
        self.device = device
        self.params = params or {}
        self.base_model = base_model
        self.le = LabelEncoder()
        self.model = None

    def _init_model(self, num_classes):
        if self.base_model is not None:
            return self.base_model

        if num_classes <= 1:
            # Trivial single class dummy
            return None

        if self.kind == "xgb":
            import xgboost
            p = {
                "n_estimators": 200,
                "max_depth": 5,
                "learning_rate": 0.05,
                "random_state": 0,
                "tree_method": "hist",
                "device": self.device,
                **self.params,
            }
            if num_classes == 2:
                p.setdefault("objective", "binary:logistic")
            else:
                p.setdefault("objective", "multi:softprob")
            return xgboost.XGBClassifier(**p)

        if self.kind == "lgbm":
            import lightgbm
            p = {
                "n_estimators": 200,
                "num_leaves": 31,
                "learning_rate": 0.05,
                "random_state": 0,
                "verbose": -1,
                **self.params,
            }
            if num_classes == 2:
                p.setdefault("objective", "binary")
            else:
                p.setdefault("objective", "multiclass")
            return lightgbm.LGBMClassifier(**p)

        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=1000, random_state=0)

    def fit(self, x: np.ndarray, y_delta: np.ndarray):
        """Fit classifier on features x and continuous residual target y_delta."""
        rounded = np.rint(y_delta).astype(int)
        clipped = np.clip(rounded, -self.max_offset, self.max_offset)
        y_encoded = self.le.fit_transform(clipped)
        num_classes = len(self.le.classes_)

        self.model = self._init_model(num_classes)
        if self.model is not None:
            self.model.fit(x, y_encoded)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict integer delta for each row."""
        if self.model is None:
            # Single class only
            return np.full(len(x), float(self.le.classes_[0]), dtype=np.float64)
        class_preds = self.model.predict(x)
        return self.le.inverse_transform(class_preds).astype(np.float64)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return class probabilities."""
        if self.model is None:
            return np.ones((len(x), 1), dtype=np.float64)
        return self.model.predict_proba(x)
