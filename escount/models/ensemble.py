"""Fast Low-Resource Collaborative Ensemble of XGBoost, LightGBM, and Ridge.

Provides Stacking, Median Voting, and Cascade blending to elevate sheet count
exact hit rate with low computational overhead (< 50ms).
"""
import numpy as np
import pandas as pd


class FastCollaborativeEnsemble:
    """Ensemble of fast regressors/classifiers (XGBoost, LightGBM, Ridge)."""

    def __init__(self, models=None, strategy="median_vote", weights=None):
        """
        Args:
            models: Dict mapping name -> fitted model instance.
            strategy: 'median_vote', 'mean_vote', or 'stacking'.
            weights: Optional dict or array of model weights for weighted voting.
        """
        self.models = models or {}
        self.strategy = strategy
        self.weights = weights
        self.meta_learner = None

    def add_model(self, name: str, model):
        self.models[name] = model

    def fit_stacking(self, oof_predictions: dict[str, np.ndarray], y_true: np.ndarray):
        """Fit a Ridge meta-learner on out-of-fold predictions."""
        from sklearn.linear_model import Ridge

        names = sorted(oof_predictions.keys())
        X_meta = np.column_stack([oof_predictions[name] for name in names])
        self.meta_learner = Ridge(alpha=1.0, positive=True).fit(X_meta, y_true)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict delta on feature matrix x using the chosen ensemble strategy."""
        if not self.models:
            raise ValueError("No base models in ensemble")

        preds = {}
        for name, model in self.models.items():
            if hasattr(model, "predict"):
                preds[name] = model.predict(x)
            else:
                raise TypeError(f"Model {name} does not have a predict method")

        if self.strategy == "stacking" and self.meta_learner is not None:
            names = sorted(preds.keys())
            X_meta = np.column_stack([preds[name] for name in names])
            return self.meta_learner.predict(X_meta)

        if self.strategy == "mean_vote":
            stack = np.column_stack(list(preds.values()))
            if self.weights is not None:
                w = np.array([self.weights.get(k, 1.0) for k in preds.keys()], dtype=np.float64)
                w = w / np.sum(w)
                return np.sum(stack * w, axis=1)
            return np.mean(stack, axis=1)

        # Default: median_vote (most robust against individual model anomalies)
        stack = np.column_stack(list(preds.values()))
        return np.median(stack, axis=1)
