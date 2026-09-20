"""PR2 edge guards (REFACTOR — locks error paths)."""
import numpy as np
import pytest

K = [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]


def test_roi_behind_camera_raises():
    from jetson.fusion.roi_manager import ROIManager

    roi = ROIManager(K, np.eye(3).tolist(), [0, 0, 0])
    with pytest.raises(ValueError):
        roi.project([0.0, 0.0, -1.0])


def test_ekf_radar_at_origin_raises():
    from jetson.fusion.ekf_tracker import EKFTracker

    ekf = EKFTracker(dt=0.1)
    with pytest.raises(ValueError):
        ekf.update_radar(1.0, origin=(0, 0, 0))  # state starts at origin


def test_ekf_initiate_bad_shape_raises():
    from jetson.fusion.ekf_tracker import EKFTracker

    with pytest.raises(ValueError):
        EKFTracker(dt=0.1).initiate([0, 0], [0.5, 0, 3])


def test_hard_rules_zero_body_and_no_r5():
    from jetson.fusion.hard_rules import gate_chain, r4_ratio_ok

    assert r4_ratio_ok(40.0, 0.0) is False
    res = gate_chain({"longest_axis_mm": 30.0, "velocity_hist": [6.0, 6.0, 6.0],
                      "thorax_v": 40.0, "band_h": 30.0, "band_s": 120.0,
                      "wingspan": 40.0, "body": 20.0})  # no sideband key
    assert res["promoted"] is True and res["r5_bonus"] == 0.0


def test_camera_unknown_class_and_bad_seconds():
    from sim.ros2.camera_sim import CameraSim

    cam = CameraSim(K=K)
    with pytest.raises(ValueError):
        cam.capture([0, 0, 2], target_class="drone")
    with pytest.raises(ValueError):
        cam.measured_fps(0.0)


def test_lidar_scan_in_band_is_float():
    from sim.gazebo.benewake_equiv.plugin import BenewakeEquiv

    r = BenewakeEquiv(seed=1).scan([0.0, 0.0, 2.0])
    assert isinstance(r, float) and 0.5 <= r <= 5.0
