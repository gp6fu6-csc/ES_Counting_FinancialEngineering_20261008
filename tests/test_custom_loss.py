"""Test mathematical properties and gradient accuracy of SoftExactHitObjective."""
import numpy as np
import pytest

from escount.models.loss import SoftExactHitObjective


def test_soft_loss_basin_and_threshold():
    obj = SoftExactHitObjective(sharpness=10.0, l2_reg=0.01)

    y_true = np.zeros(5)
    # Inside basin |e| <= 0.3 -> loss should be very small
    y_pred_safe = np.array([0.0, 0.1, -0.1, 0.2, -0.2])
    loss_safe = obj.loss(y_true, y_pred_safe)

    # At threshold |e| = 0.5 -> loss should be around 0.5 + reg
    y_pred_thresh = np.array([0.5, -0.5, 0.5, -0.5, 0.5])
    loss_thresh = obj.loss(y_true, y_pred_thresh)

    # Far miss |e| = 1.0 -> loss should be close to 1.0 + reg
    y_pred_miss = np.array([1.0, -1.0, 1.2, -1.2, 1.5])
    loss_miss = obj.loss(y_true, y_pred_miss)

    assert np.all(loss_safe < loss_thresh)
    assert np.all(loss_thresh < loss_miss)
    assert np.all(loss_safe < 0.2)
    assert np.all(loss_thresh > 0.45)


def test_gradient_numerical_finite_difference():
    obj = SoftExactHitObjective(sharpness=8.0, eps=1e-3, l2_reg=0.05)
    delta = 1e-6

    y_true = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    y_pred = np.array([-0.8, -0.49, -0.1, 0.05, 0.51, 0.95])

    grad_analytical, hess_analytical = obj.gradient_and_hessian(y_true, y_pred)

    # Numerical grad = (L(y_pred + delta) - L(y_pred - delta)) / (2 * delta)
    loss_plus = obj.loss(y_true, y_pred + delta)
    loss_minus = obj.loss(y_true, y_pred - delta)
    grad_numerical = (loss_plus - loss_minus) / (2.0 * delta)

    np.testing.assert_allclose(grad_analytical, grad_numerical, rtol=1e-3, atol=1e-4)

    # Ensure Hessian is strictly positive everywhere
    assert np.all(hess_analytical > 0.0)
