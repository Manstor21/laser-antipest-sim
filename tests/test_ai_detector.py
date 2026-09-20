"""RED 2.4a — PR2 detector ROI-only 320 ONNX stub (falla sin detector.py)."""
import numpy as np


def test_predict_roi_only_320_returns_mask_and_probs():
    from jetson.ai import detector

    crop = np.zeros((320, 320, 3), dtype=np.uint8)
    crop[150:170, 150:170, 0] = 200  # fake thorax blob
    out = detector.predict(crop)
    assert "mask" in out and "p_velutina" in out and "p_bee" in out, (
        f"predict must return mask+Pv+Pb, got {sorted(out)}"
    )
    mask = out["mask"]
    assert mask.shape == (320, 320), f"mask must be 320x320, got {mask.shape}"
    assert 0.0 <= float(out["p_velutina"]) <= 1.0
    assert 0.0 <= float(out["p_bee"]) <= 1.0


def test_predict_rejects_non_roi_input():
    from jetson.ai import detector

    import pytest

    full = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        detector.predict(full)
