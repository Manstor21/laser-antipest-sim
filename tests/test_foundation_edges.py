"""REFACTOR — edge guards for PR1 foundation (push coverage >=90%)."""
import numpy as np
import pytest


def test_sync_empty_series_raises():
    import sim.ros2.sync as s

    with pytest.raises(ValueError):
        s.compute_sync_rms_ms([])


def test_sync_projection_behind_camera_raises():
    import sim.ros2.sync as s

    K = np.eye(3)
    R = np.eye(3)
    t = np.zeros(3)
    with pytest.raises(ValueError):
        s.project_points([[0.0, 0.0, -1.0]], K, R, t)


def test_sync_shape_mismatch_raises():
    import sim.ros2.sync as s

    with pytest.raises(ValueError):
        s.reprojection_rmse_px(np.zeros((2, 2)), np.zeros((3, 2)))


def test_mic_stub_read_without_connect_raises():
    from jetson.sensors.mems_mic.stub import MicStub

    with pytest.raises(RuntimeError):
        MicStub().read()
