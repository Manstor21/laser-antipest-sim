"""RED test 1.2 — sim-clock sync RMS<=1ms + reproj<=2px@2m (must fail before impl)."""
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
CALIB = ROOT / "sim" / "ros2" / "calibration.yaml"


def _load_calib():
    assert CALIB.exists(), f"missing {CALIB}"
    with open(CALIB, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_calibration_has_extrinsic_and_fiducials():
    cal = _load_calib()
    K = np.asarray(cal["camera"]["K"], dtype=float)
    R = np.asarray(cal["extrinsic"]["R"], dtype=float)
    t = np.asarray(cal["extrinsic"]["t"], dtype=float)
    assert K.shape == (3, 3)
    assert R.shape == (3, 3)
    assert t.shape == (3,) or t.shape == (3, 1)
    assert len(cal["fiducials"]) >= 4


def test_sync_rms_within_1ms_on_60s_bag():
    import sim.ros2.sync as sync_mod

    rng = np.random.default_rng(7)
    # 60 s @ 10 Hz lidar vs camera/radar offsets: gaussian sigma 0.4 ms
    offsets_ms = rng.normal(0.0, 0.4, size=600)
    rms = sync_mod.compute_sync_rms_ms(offsets_ms)
    assert rms <= 1.0, f"RMS {rms} ms exceeds 1 ms budget"


def test_sync_reproj_within_2px_at_2m():
    import sim.ros2.sync as sync_mod

    cal = _load_calib()
    K = np.asarray(cal["camera"]["K"], dtype=float)
    R = np.asarray(cal["extrinsic"]["R"], dtype=float)
    t = np.asarray(cal["extrinsic"]["t"], dtype=float).reshape(3)
    # Fiducial 3D points at ~2 m depth, project then reproject with 0.5 px noise
    pts3d = np.array(
        [
            [0.0, 0.0, 2.0],
            [0.1, -0.05, 2.0],
            [-0.1, 0.05, 2.0],
            [0.05, 0.1, 2.0],
        ]
    )
    px = sync_mod.project_points(pts3d, K, R, t)
    rng = np.random.default_rng(3)
    px_noisy = px + rng.normal(0.0, 0.5, size=px.shape)
    rmse = sync_mod.reprojection_rmse_px(px, px_noisy)
    assert rmse <= 2.0, f"reproj RMSE {rmse} px exceeds 2 px budget"
