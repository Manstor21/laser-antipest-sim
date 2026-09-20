"""RED 2.2 — EKF predict_at(t_fire,t_meas) + fusion t_meas (PR2)."""
import numpy as np


def test_predict_at_uses_dt_50ms():
    from jetson.fusion.ekf_tracker import EKFTracker

    ekf = EKFTracker(dt=0.1)
    ekf.x = np.array([0, 0, 3, 5, 0, 0, 0, 0, 0], dtype=float)
    pred = ekf.predict_at(t_fire=0.05, t_meas=0.0)
    assert abs(float(pred[0]) - 0.25) < 1e-9
    assert abs(float(pred[2]) - 3.0) < 1e-9


def test_predict_at_triangulate_100ms():
    from jetson.fusion.ekf_tracker import EKFTracker

    ekf = EKFTracker(dt=0.1)
    ekf.x = np.array([1.0, 0, 3, 5, 0, 0, 0, 0, 0], dtype=float)
    before = ekf.x.copy()
    pred = ekf.predict_at(t_fire=0.2, t_meas=0.1)
    assert abs(float(pred[0]) - 1.5) < 1e-9
    # non-mutating: state unchanged
    assert np.allclose(ekf.x, before)


def test_predict_ms_delegates_to_predict_at():
    from jetson.fusion.ekf_tracker import EKFTracker

    ekf = EKFTracker(dt=0.1)
    ekf.x = np.array([0, 0, 3, 2.0, 0, 0, 0, 0, 0], dtype=float)
    a = ekf.predict_ms(50)
    b = ekf.predict_at(t_fire=0.05, t_meas=0.0)
    assert np.allclose(a, b)


def test_fusion_step_propagates_t_meas():
    import numpy as np

    from jetson.fusion.fusion_pipeline import FusionPipeline

    K = [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]
    R = np.eye(3).tolist()
    t = [0, 0, 0]
    pipe = FusionPipeline(K, R, t, dt=0.1)
    pipe.initiate([0.0, 0.0, 3.0], [0.5, 0.0, 3.0])
    out = pipe.step(lidar_xyz=[0.6, 0.0, 3.0], t_meas=1.0, t_fire=1.05)
    assert out["t_meas"] == 1.0
    assert out["t_fire"] == 1.05
    assert len(out["x9"]) == 9
