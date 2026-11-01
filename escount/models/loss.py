"""Custom Loss Functions for Industrial Exact Hit Optimization.

Provides Soft Exact-Hit Loss for XGBoost and LightGBM with analytical first derivative
(grad) and positive-definite second derivative (hess).
"""
import numpy as np


class SoftExactHitObjective:
    """Soft Exact-Hit objective function for gradient boosting (XGBoost / LightGBM).

    Penalizes errors near and beyond the rounding threshold (|e| >= 0.5), with minimal
    gradient inside the safe basin (|e| <= 0.4) and positive-definite regularization.
    """

    def __init__(self, sharpness: float = 8.0, eps: float = 1e-3, l2_reg: float = 0.05):
        """
        Args:
            sharpness: Steepness of the sigmoid transition around |e| = 0.5 (default: 8.0).
            eps: Smoothing epsilon for smooth absolute value sqrt(e^2 + eps^2).
            l2_reg: Global L2 regularization ensuring strictly positive Hessian (default: 0.05).
        """
        self.sharpness = float(sharpness)
        self.eps = float(eps)
        self.l2_reg = float(l2_reg)

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute the point-wise loss."""
        e = y_pred - y_true
        r = np.sqrt(e ** 2 + self.eps ** 2)
        z = self.sharpness * (r - 0.5)
        # Numerically stable sigmoid
        sig = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
        return sig + 0.5 * self.l2_reg * (e ** 2)

    def gradient_and_hessian(self, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute analytical (grad, hess) with respect to y_pred."""
        e = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
        r2 = e ** 2 + self.eps ** 2
        r = np.sqrt(r2)

        z = self.sharpness * (r - 0.5)
        z_clipped = np.clip(z, -30.0, 30.0)
        sig = 1.0 / (1.0 + np.exp(-z_clipped))
        dsig = sig * (1.0 - sig)                      # sigma'
        d2sig = dsig * (1.0 - 2.0 * sig)              # sigma''

        dr_de = e / r
        d2r_de2 = (self.eps ** 2) / (r2 * r)

        k = self.sharpness
        d_sig_de = k * dsig * dr_de
        d2_sig_de2 = k * (k * d2sig * (dr_de ** 2) + dsig * d2r_de2)

        grad = d_sig_de + self.l2_reg * e
        hess = d2_sig_de2 + self.l2_reg

        # Guarantee strictly positive Hessian for XGBoost/LightGBM tree builders
        hess = np.clip(hess, 1e-4, 100.0)
        return grad, hess

    def xgb_objective(self, y_true, y_pred):
        """Callable signature for XGBoost custom objective."""
        return self.gradient_and_hessian(y_true, y_pred)

    def lgb_objective(self, y_true, y_pred):
        """Callable signature for LightGBM custom objective (preds, dataset)."""
        # LightGBM passes (preds, dataset)
        labels = y_true.get_label() if hasattr(y_true, "get_label") else y_true
        return self.gradient_and_hessian(labels, y_pred)
