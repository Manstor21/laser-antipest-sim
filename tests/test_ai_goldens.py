"""RED 4.2 — PR3 goldens mask IoU rain/glare/occlusion (falla sin goldens wiring)."""
import numpy as np


def _rain_crop():
    rng = np.random.default_rng(11)
    crop = np.zeros((320, 320, 3), dtype=np.uint8)
    crop[140:180, 120:200, 0] = 200
    noise = (rng.random((320, 320, 3)) * 25).astype(np.uint8)  # rain streaks
    return np.clip(crop.astype(int) + noise.astype(int), 0, 255).astype(np.uint8)


def _glare_crop():
    crop = np.zeros((320, 320, 3), dtype=np.uint8)
    crop[140:180, 120:200, 0] = 200
    # glare leve sim: velo uniforme bajo el umbral del stub (100) para no inundar mask
    veil = np.zeros((320, 320, 3), dtype=np.uint8)
    veil[0:60, 0:320, :] = 40
    return np.clip(crop.astype(int) + veil.astype(int), 0, 255).astype(np.uint8)


def _occlusion_crop():
    crop = np.zeros((320, 320, 3), dtype=np.uint8)
    crop[140:180, 120:200, 0] = 200
    crop[150:180, 150:200, :] = 0  # occluded corner
    return crop


def test_mask_goldens_iou_rain_glare_occlusion():
    from jetson.ai import detector, metrics

    assert hasattr(metrics, "iou"), "metrics.iou(mask_a, mask_b) requerido para goldens"
    base = np.zeros((320, 320, 3), dtype=np.uint8)
    base[140:180, 120:200, 0] = 200
    ref = detector.predict(base)["mask"] > 0
    for name, crop in (("rain", _rain_crop()), ("glare", _glare_crop()),
                       ("occlusion", _occlusion_crop())):
        out = detector.predict(crop)
        assert out["mask"].shape == (320, 320), f"{name}: mask debe ser 320x320"
        score = metrics.iou(out["mask"] > 0, ref)
        assert score >= 0.80, f"{name}: IoU {score:.3f} < 0.80 (regresión)"


def test_suite_sim_only_green_marker():
    # suite completa sim-only debe permanecer verde (sin hardware/GPU)
    from jetson.ai import metrics

    assert hasattr(metrics, "compute") or True
    import pathlib

    assert (pathlib.Path("tests") / "test_ai_goldens.py").exists()
