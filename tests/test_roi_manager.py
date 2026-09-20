"""RED test 2.2 — ROI 320x320 + fallback (must fail before impl)."""
import numpy as np

from jetson.fusion.roi_manager import ROIManager

K = [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]
R = np.eye(3).tolist()
T = [0.0, 0.0, 0.0]


def test_projection_center_at_2m():
    roi = ROIManager(K, R, T)
    u, v = roi.project([0.0, 0.0, 2.0])
    assert abs(u - 320.0) < 1e-9
    assert abs(v - 240.0) < 1e-9


def test_roi_contains_target_at_2m():
    roi = ROIManager(K, R, T, roi_size=320)
    box = roi.get_roi([0.0, 0.0, 2.0])
    assert box["w"] == 320 and box["h"] == 320
    assert box["fallback"] is False
    assert roi.contains(box, *roi.project([0.0, 0.0, 2.0]))


def test_fallback_after_6_misses():
    roi = ROIManager(K, R, T, miss_fallback_frames=5)
    box = None
    for _ in range(6):
        box = roi.update(found=False, track_xyz=[0.0, 0.0, 2.0])
    assert box["fallback"] is True
    assert box["w"] == 640 and box["h"] == 480  # full-frame re-acquire


def test_reacquire_resets_miss_counter():
    roi = ROIManager(K, R, T, miss_fallback_frames=5)
    for _ in range(4):
        roi.update(found=False, track_xyz=[0.0, 0.0, 2.0])
    box = roi.update(found=True, track_xyz=[0.0, 0.0, 2.0])
    assert box["fallback"] is False
    assert roi.misses == 0


def test_roi_clamped_inside_image():
    roi = ROIManager(K, R, T, roi_size=320)
    box = roi.get_roi([2.0, 0.0, 2.0])  # projects far right (u=1520)
    assert 0 <= box["x"] <= 640 - 320
    assert 0 <= box["y"] <= 480 - 320
