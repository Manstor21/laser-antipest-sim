"""Sim-clock sync helpers (ROS2 sim-time, zero hardware).

Verifies FUSION-Sync-Calib: RMS <= 1 ms over 60 s bag + reproj <= 2 px @ 2 m.
Pure numpy so pytest runs without a ROS2 installation.
"""

import numpy as np


def compute_sync_rms_ms(offsets_ms) -> float:
    """Root-mean-square of per-message clock offsets in milliseconds."""
    arr = np.asarray(offsets_ms, dtype=float)
    if arr.size == 0:
        raise ValueError("empty offset series")
    return float(np.sqrt(np.mean(arr**2)))


def project_points(points_xyz, K, R, t) -> np.ndarray:
    """Project 3D points (Nx3, sim frame) to pixels via K·[R|t]."""
    pts = np.asarray(points_xyz, dtype=float)
    K = np.asarray(K, dtype=float)
    R = np.asarray(R, dtype=float)
    t = np.asarray(t, dtype=float).reshape(3)
    cam = (R @ pts.T).T + t
    z = cam[:, 2]
    if np.any(z <= 0):
        raise ValueError("points must be in front of camera (z > 0)")
    homo = (K @ cam.T).T
    return homo[:, :2] / z[:, None]


def reprojection_rmse_px(px_ref, px_meas) -> float:
    """RMSE in pixels between reference and measured pixel sets."""
    a = np.asarray(px_ref, dtype=float)
    b = np.asarray(px_meas, dtype=float)
    if a.shape != b.shape:
        raise ValueError("pixel sets must share shape")
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))
