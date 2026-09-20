"""Benewake-equiv LiDAR sim plugin (0.5-5 m, >=10 Hz, sim-only).

Single-target range model with gaussian noise for Gazebo-free pytest;
the Gazebo wrapper publishes /velutina/lidar at sim rate in PR3 wiring.
Out-of-range targets (>5 m, <0.5 m) yield no cluster (None).
"""

import numpy as np


class BenewakeEquiv:
    """Sparse short-range LiDAR equivalent for insect-size gating."""

    def __init__(self, range_min_m=0.5, range_max_m=5.0, rate_hz=10,
                 noise_mm=15.0, seed=0):
        self.range_min = float(range_min_m)
        self.range_max = float(range_max_m)
        self.rate_hz = int(rate_hz)
        self.noise_m = float(noise_mm) / 1000.0
        self.rng = np.random.default_rng(seed)

    def scan(self, target_xyz, origin=(0.0, 0.0, 0.0)):
        """One sim scan: noisy range in metres, or None if out of band."""
        d = np.asarray(target_xyz, dtype=float).reshape(3) - \
            np.asarray(origin, dtype=float).reshape(3)
        true = float(np.linalg.norm(d))
        if not self.range_min <= true <= self.range_max:
            return None
        return true + float(self.rng.normal(0.0, self.noise_m))
