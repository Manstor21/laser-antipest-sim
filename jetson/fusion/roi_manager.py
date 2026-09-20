"""Dynamic 320x320 ROI manager (sim extrinsic, zero hardware).

Projects EKF tracks via K·[R|t], crops a 320x320 box clamped to the
640x480 sim frame, and falls back to full-frame after >5 missed frames.
Pure numpy so pytest runs without a ROS2 installation.
"""

import numpy as np


class ROIManager:
    """Project tracks to pixels and own the ROI / fallback state."""

    def __init__(self, K, R, t, img_w=640, img_h=480, roi_size=320,
                 miss_fallback_frames=5):
        self.K = np.asarray(K, dtype=float).reshape(3, 3)
        self.R = np.asarray(R, dtype=float).reshape(3, 3)
        self.t = np.asarray(t, dtype=float).reshape(3)
        self.img_w = int(img_w)
        self.img_h = int(img_h)
        self.roi_size = int(roi_size)
        self.miss_fallback_frames = int(miss_fallback_frames)
        self.misses = 0

    def project(self, xyz):
        """3D sim point → pixel (u, v) via K·(R·p + t)."""
        p = np.asarray(xyz, dtype=float).reshape(3)
        cam = self.R @ p + self.t
        if cam[2] <= 0:
            raise ValueError("point must be in front of camera (z > 0)")
        homo = self.K @ cam
        return float(homo[0] / cam[2]), float(homo[1] / cam[2])

    def get_roi(self, xyz):
        """320x320 box centered on the projection, clamped to frame."""
        u, v = self.project(xyz)
        half = self.roi_size // 2
        x = min(max(int(round(u - half)), 0), self.img_w - self.roi_size)
        y = min(max(int(round(v - half)), 0), self.img_h - self.roi_size)
        return {"x": x, "y": y, "w": self.roi_size, "h": self.roi_size,
                "u": u, "v": v, "fallback": False}

    @staticmethod
    def contains(box, u, v):
        """True if pixel (u, v) lies inside box."""
        return box["x"] <= u <= box["x"] + box["w"] and \
            box["y"] <= v <= box["y"] + box["h"]

    def update(self, found, track_xyz):
        """Track-loss state machine; full-frame after >N misses."""
        if found:
            self.misses = 0
            return self.get_roi(track_xyz)
        self.misses += 1
        if self.misses > self.miss_fallback_frames:
            u, v = self.project(track_xyz)
            return {"x": 0, "y": 0, "w": self.img_w, "h": self.img_h,
                    "u": u, "v": v, "fallback": True}
        return self.get_roi(track_xyz)
