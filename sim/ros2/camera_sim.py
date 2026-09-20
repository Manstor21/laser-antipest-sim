"""Camera sim: 120 fps projection + synthetic R3 HSV (sim-only).

Projects the sim target via K (identity extrinsic) and emits the HSV
features the R3 gate consumes: velutina (dark thorax + orange band) vs
bee (amber-uniform, R3 fail). Frame counter proves the 120 fps rate.
"""

import numpy as np

# Synthetic R3 feature packs per target class.
FEATURES = {
    "velutina": {"thorax_v": 40.0, "band_h": 30.0, "band_s": 120.0},
    "bee": {"thorax_v": 150.0, "band_h": 30.0, "band_s": 40.0},
    "leaf": {"thorax_v": 110.0, "band_h": 70.0, "band_s": 60.0},
}


class CameraSim:
    """Pinhole camera equivalent feeding ROI/R3 without Gazebo."""

    def __init__(self, K, img_w=640, img_h=480, fps=120):
        self.K = np.asarray(K, dtype=float).reshape(3, 3)
        self.img_w = int(img_w)
        self.img_h = int(img_h)
        self.fps = int(fps)
        self.frames = 0

    def project(self, xyz):
        """3D sim point → pixel (u, v). Identity extrinsic sim camera."""
        p = np.asarray(xyz, dtype=float).reshape(3)
        if p[2] <= 0:
            raise ValueError("point must be in front of camera (z > 0)")
        homo = self.K @ p
        return float(homo[0] / p[2]), float(homo[1] / p[2])

    def capture(self, xyz, target_class="velutina"):
        """One sim frame: pixels + class-driven HSV for R3."""
        u, v = self.project(xyz)
        try:
            hsv = FEATURES[target_class]
        except KeyError:
            raise ValueError(f"unknown target_class {target_class!r}")
        self.frames += 1
        return {"u": u, "v": v, "frame": self.frames,
                "target_class": target_class, **hsv}

    def measured_fps(self, sim_seconds):
        """Frame count over elapsed sim seconds."""
        if sim_seconds <= 0:
            raise ValueError("sim_seconds must be positive")
        return self.frames / float(sim_seconds)
