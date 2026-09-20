"""60 GHz radar sim: radial velocity + wingbeat sidebands (sim-only).

Publishes /velutina/radar equivalents for pytest without ROS2: hover
(|v|<=0.5) vs transit (1-11 m/s) labels plus an R5 advisory priority
that never rejects alone. Mirrors config.yaml R2/R5 thresholds.
"""

import numpy as np


class RadarSim:
    """Noisy Doppler observer with micro-Doppler advisory channel."""

    def __init__(self, vel_noise=0.1, seed=0):
        self.vel_noise = float(vel_noise)
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def label(v_true):
        """Hover / transit / invalid banding per R2."""
        a = abs(float(v_true))
        if a <= 0.5:
            return "hover"
        if 1.0 <= a <= 11.0:
            return "transit"
        return "invalid"

    def observe(self, v_true, sideband_hz=0.0):
        """One sim dwell: measured velocity + advisory priority."""
        v_meas = float(v_true) + float(self.rng.normal(0.0, self.vel_noise))
        try:
            f = abs(float(sideband_hz))
        except (TypeError, ValueError):
            f = 0.0
        has_band = 100.0 <= f <= 170.0
        return {"v_true": float(v_true), "v_meas": v_meas,
                "label": self.label(v_true), "sideband_hz": f,
                "priority": 0.8 if has_band else 0.5,
                "rejected_by_r5": False}
