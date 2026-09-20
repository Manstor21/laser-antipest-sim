"""Tests del backend YOLO (jetson/ai/yolo_backend.py). Rapidos (<60s), SIN las 380 fotos."""

from __future__ import annotations

import numpy as np
import pytest

from jetson.ai import yolo_backend as yb

ultralytics = pytest.importorskip("ultralytics", reason="sin ultralytics (pip install ultralytics)")

needs_pt = pytest.mark.skipif(
    not yb.MODEL_PATH.exists(), reason="sin peso YOLO (copia DESCARGAS/best.pt)"
)


def _synthetic_crop(seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(320, 320, 3), dtype=np.uint8)


def test_pt_existe_y_carga():
    assert yb.MODEL_PATH.exists(), f"falta {yb.MODEL_PATH}"
    m = yb.load()
    assert "velutina" in {v for v in m["model"].names.values()}
    assert m["idx_velutina"] >= 0


@needs_pt
def test_proba_en_rango_y_suma():
    pv, pb = yb.predict_proba(_synthetic_crop())
    assert 0.0 <= pv <= 1.0
    assert 0.0 <= pb <= 1.0
    assert pb == pytest.approx(1.0 - pv)


@needs_pt
def test_determinista():
    crop = _synthetic_crop(7)
    a = yb.predict_proba(crop)[0]
    b = yb.predict_proba(crop)[0]
    assert a == pytest.approx(b)


def test_crop_no320_rechaza():
    bad = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="320x320x3"):
        yb.predict_proba(bad)


def test_load_sin_pt_error_claro(monkeypatch, tmp_path):
    monkeypatch.setattr(yb, "MODEL_PATH", tmp_path / "no_existe.pt")
    yb.reset_cache()
    try:
        with pytest.raises(FileNotFoundError):
            yb.load()
    finally:
        yb.reset_cache()


V2_PATH = yb.REPO_ROOT / "jetson" / "ai" / "yolov8s_velutina_v2.pt"

needs_v2 = pytest.mark.skipif(
    not V2_PATH.exists(), reason="sin peso v2 (copia DESCARGAS/best.pt)"
)


@needs_v2
def test_v2_existe_y_carga():
    from ultralytics import YOLO

    assert V2_PATH.exists(), f"falta {V2_PATH}"
    assert V2_PATH.stat().st_size < 20 * 1024 * 1024, "v2 debe pesar <20MB"
    m = YOLO(str(V2_PATH))
    assert "velutina" in {v for v in m.names.values()}
    assert len(m.names) == 7  # abeja/bombus/crabro/eristalis/germanica/velutina/vespula


@needs_v2
def test_backend_v2_proba_en_rango(monkeypatch):
    monkeypatch.setattr(yb, "MODEL_PATH", V2_PATH)
    yb.reset_cache()
    try:
        pv, pb = yb.predict_proba(_synthetic_crop(3))
    finally:
        yb.reset_cache()
    assert 0.0 <= pv <= 1.0
    assert 0.0 <= pb <= 1.0
    assert pb == pytest.approx(1.0 - pv)
