"""Test DiscreteResidualClassifier integer mapping and prediction."""
import numpy as np
import pytest

from escount.models.classifier import DiscreteResidualClassifier


def test_classifier_fit_predict():
    np.random.seed(42)
    n_samples = 100
    n_features = 8
    X = np.random.randn(n_samples, n_features)

    # True delta: mostly 0, +1, or -1 with some noise
    y_true_delta = np.random.choice([-1.0, 0.0, 1.0], size=n_samples, p=[0.25, 0.5, 0.25])
    # Add small sub-sheet continuous noise
    y_noisy = y_true_delta + np.random.uniform(-0.2, 0.2, size=n_samples)

    clf = DiscreteResidualClassifier(kind="xgb", max_offset=3, device="cpu", params={"n_estimators": 20})
    clf.fit(X, y_noisy)

    preds = clf.predict(X)
    assert len(preds) == n_samples
    # All predictions must be exact integers!
    assert np.all(np.equal(np.mod(preds, 1), 0))
    # Must lie within [-3, 3]
    assert np.all(preds >= -3) and np.all(preds <= 3)
