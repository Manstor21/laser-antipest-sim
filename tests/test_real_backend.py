"""Tests del backend REAL (jetson/ai/real_backend.py). Rapidos (<60s), SIN las 380 fotos."""

from __future__ import annotations

import numpy as np
import pytest

from jetson.ai import real_backend as rb

EXPECTED_NAMES = (
    [f"H{i:02d}" for i in range(16)]
    + [f"S{i}" for i in range(8)]
    + [f"V{i}" for i in range(8)]
    + ["dark_ratio", "bright_ratio", "edge_density"]
)


def _dark_crop():
    return np.full((320, 320, 3), 20, dtype=np.uint8)


def _bright_crop():
    return np.full((320, 320, 3), 230, dtype=np.uint8)


def test_feature_order_stable():
    assert rb.FEATURE_NAMES == EXPECTED_NAMES
    assert len(rb.FEATURE_NAMES) == 35


def test_features_synthetic_shapes_and_ranges():
    for crop in (_dark_crop(), _bright_crop()):
        f = rb.extract_features(crop)
        assert f.shape == (35,)
        assert np.all(np.isfinite(f))
    fd, fb = rb.extract_features(_dark_crop()), rb.extract_features(_bright_crop())
    i_dark = rb.FEATURE_NAMES.index("dark_ratio")
    i_bright = rb.FEATURE_NAMES.index("bright_ratio")
    assert fd[i_dark] > fb[i_dark]  # el crop oscuro es mas oscuro
    assert fb[i_bright] > fd[i_bright]  # el crop claro es mas claro
    assert abs(fd[:32].sum() - 3.0) < 1e-6  # 3 histogramas normalizados


def test_sigmoid_monotona():
    assert rb._sigmoid(-2.0) < rb._sigmoid(0.0) < rb._sigmoid(2.0)
    assert abs(rb._sigmoid(0.0) - 0.5) < 1e-12


def test_dark_mayor_pv_dummy_weights(monkeypatch):
    """Sin el npz: con pesos dummy (dark_ratio positivo) mas oscuro -> mayor Pv."""
    monkeypatch.setattr(rb, "_cache", {
        "coef": np.array([0.0] * 32 + [5.0, -5.0, 0.0]),
        "intercept": 0.0,
        "mean": np.zeros(35),
        "scale": np.ones(35),
        "threshold_fp0": 0.9,
        "path": "dummy",
    })
    pd, pbd = rb.predict_proba(_dark_crop())
    pb, pbb = rb.predict_proba(_bright_crop())
    assert pd > pb
    assert pbd == pytest.approx(1.0 - pd)
    assert pbb == pytest.approx(1.0 - pb)


def test_load_sin_npz_error_claro(monkeypatch, tmp_path):
    monkeypatch.setattr(rb, "MODEL_PATH", tmp_path / "no_existe.npz")
    rb.reset_cache()
    try:
        with pytest.raises(FileNotFoundError, match="train_real"):
            rb.load()
    finally:
        rb.reset_cache()


def test_backend_train_feature_parity():
    """Las features del backend son identicas a las del train (mismo crop)."""
    train = pytest.importorskip("sim.dataset.train_real")
    rng = np.random.default_rng(0)
    crop = rng.integers(0, 256, size=(320, 320, 3), dtype=np.uint8)
    assert np.allclose(train.extract_features(crop), rb.extract_features(crop), atol=1e-12)


needs_npz = pytest.mark.skipif(
    not rb.MODEL_PATH.exists(), reason="sin npz real (entrena con train_real.py)"
)


@needs_npz
def test_integracion_npz_real():
    m = rb.load()
    assert m["coef"].shape == (35,) and m["mean"].shape == (35,) and m["scale"].shape == (35,)
    assert 0.0 < m["threshold_fp0"] <= 1.0
    pd, pbd = rb.predict_proba(_dark_crop())
    pb, pbb = rb.predict_proba(_bright_crop())
    for p in (pd, pbd, pb, pbb):
        assert 0.0 <= p <= 1.0
    assert pbd == pytest.approx(1.0 - pd)
    assert pd == pytest.approx(rb.predict_proba(_dark_crop())[0])  # determinista
    assert pd > pb  # mas oscuro -> mayor Pv (verificado con los pesos reales)
    # Recomputacion manual con la misma formula.
    feats = rb.extract_features(_dark_crop())
    z = (feats - m["mean"]) / m["scale"]
    logit = float(m["coef"] @ z + m["intercept"])
    assert pd == pytest.approx(1.0 / (1.0 + np.exp(-logit)))
