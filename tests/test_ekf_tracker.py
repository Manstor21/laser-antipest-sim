"""RED test 2.1 — EKF 9-DOF + 50ms predict (must fail before impl)."""
import numpy as np

from jetson.fusion.ekf_tracker import EKFTracker


def test_state_is_9dof():
    ekf = EKFTracker(dt=0.1)
    assert ekf.x.shape == (9,)
    assert ekf.P.shape == (9, 9)


def test_predict_50ms_constant_velocity():
    ekf = EKFTracker(dt=0.1)
    ekf.x = np.array([0, 0, 3, 5, 0, 0, 0, 0, 0], dtype=float)
    pred = ekf.predict_ms(50)
    assert abs(pred[0] - 0.25) < 1e-9  # 5 m/s * 0.05 s
    assert abs(pred[2] - 3.0) < 1e-9


def test_initiate_from_two_fixes():
    ekf = EKFTracker(dt=0.1)
    ekf.initiate([0.0, 0.0, 3.0], [0.5, 0.0, 3.0])
    assert abs(ekf.x[3] - 5.0) < 1e-9
    assert ekf.P.shape == (9, 9)


def test_latency_compensation_5ms_crossing():
    rng = np.random.default_rng(7)
    rng_radar = np.random.default_rng(18)
    dt = 0.1
    ekf = EKFTracker(dt=dt)
    # Straight 5 m/s crossing at 3 m depth, x in [-4, +4] m so the
    # range stays inside the 0.5-5 m Benewake-equiv band.
    t = -0.8
    z0 = np.array([5 * t, 0.0, 3.0]) + rng.normal(0, 0.010, 3)
    t += dt
    z1 = np.array([5 * t, 0.0, 3.0]) + rng.normal(0, 0.010, 3)
    ekf.initiate(z0, z1)
    errs = []
    for _ in range(14):
        t += dt
        true = np.array([5 * t, 0.0, 3.0])
        assert float(np.linalg.norm(true)) <= 5.0  # in lidar band
        ekf.predict()
        ekf.update_lidar(true + rng.normal(0, 0.010, 3))  # 10 mm fixes
        vr = float(5 * (5 * t) / np.linalg.norm(true))
        ekf.update_radar(vr + rng_radar.normal(0, 0.1))  # Doppler aid
        # Emulate 40 ms delayed estimate, compensate with +50 ms predict.
        pred = ekf.predict_ms(50)
        future_true = np.array([5 * (t + 0.05), 0.0, 3.0])
        errs.append(float(np.linalg.norm(pred[:3] - future_true)))
    # Skip convergence transient, RMS must be < 15 mm.
    arr = np.asarray(errs[5:])
    rms = float(np.sqrt(np.mean(arr ** 2)))
    assert rms < 0.015, f"RMS {rms * 1000:.1f} mm >= 15 mm"


def test_radar_update_rejects_bad_shape():
    import pytest

    ekf = EKFTracker(dt=0.1)
    with pytest.raises(ValueError):
        ekf.update_lidar([1.0, 2.0])  # needs xyz
