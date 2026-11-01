"""Test FastCollaborativeEnsemble voting and stacking."""
import numpy as np
import pytest

from escount.models.ensemble import FastCollaborativeEnsemble


class DummyModel:
    def __init__(self, offset):
        self.offset = offset

    def predict(self, x):
        return np.full(len(x), self.offset, dtype=np.float64)


def test_ensemble_voting():
    m1 = DummyModel(1.0)
    m2 = DummyModel(2.0)
    m3 = DummyModel(3.0)

    ens = FastCollaborativeEnsemble(
        models={"m1": m1, "m2": m2, "m3": m3},
        strategy="median_vote"
    )
    X = np.zeros((10, 5))
    preds = ens.predict(X)
    assert np.allclose(preds, 2.0)

    ens_mean = FastCollaborativeEnsemble(
        models={"m1": m1, "m2": m2, "m3": m3},
        strategy="mean_vote"
    )
    assert np.allclose(ens_mean.predict(X), 2.0)


def test_ensemble_stacking():
    np.random.seed(42)
    n = 50
    y_true = np.random.uniform(-1, 1, size=n)
    oof1 = y_true + np.random.normal(0, 0.1, size=n)
    oof2 = y_true + np.random.normal(0, 0.2, size=n)

    m1 = DummyModel(0.5)
    m2 = DummyModel(0.5)
    ens = FastCollaborativeEnsemble(models={"m1": m1, "m2": m2}, strategy="stacking")
    ens.fit_stacking({"m1": oof1, "m2": oof2}, y_true)

    X = np.zeros((5, 3))
    preds = ens.predict(X)
    assert len(preds) == 5
