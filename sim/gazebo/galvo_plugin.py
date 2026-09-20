"""Gazebo galvo plugin sim (SIMULATED, 0EUR, no HW).

Projects EKF target via K*[R|t] geometry to (theta_x, theta_y), drives
GalvoSim, evaluates f-theta fluence, aborts when error >10mm.
Supports `python sim/gazebo/galvo_plugin.py --sim-step` harness.
"""

import argparse
import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from firmware.galvo_shutter.galvo_sim import ABORT_ERROR_MM, GalvoSim
from sim.gazebo.f_theta_optics import evaluate_shot

SIMULATED = True
FOCAL_MM = 100.0


def project_angles(target_xyz, K=None, R=None, t=None):
    """Angles from camera-frame target: theta = atan(x/z), atan(y/z).

    Applies optional [R|t] extrinsics then K-compatible pinhole geometry.
    K scales pixel projection only; angles use normalized coords.
    """
    p = np.asarray(target_xyz, dtype=float).reshape(3)
    Rm = np.eye(3) if R is None else np.asarray(R, dtype=float).reshape(3, 3)
    tv = np.zeros(3) if t is None else np.asarray(t, dtype=float).reshape(3)
    pc = Rm @ p + tv
    if pc[2] <= 0:
        raise ValueError("target behind camera (z<=0)")
    theta_x = math.atan2(pc[0], pc[2])
    theta_y = math.atan2(pc[1], pc[2])
    return float(theta_x), float(theta_y)


def angles_to_field_mm(theta_x, theta_y, focal_mm=FOCAL_MM):
    """f-theta field position: x = f*theta (paraxial)."""
    return float(focal_mm * theta_x), float(focal_mm * theta_y)


class GalvoPlugin:
    """EKF target -> galvo angles -> settle/abort -> fluence log."""

    def __init__(self, K=None, R=None, t=None, galvo=None):
        self.K = None if K is None else np.asarray(K, dtype=float)
        self.R = None if R is None else np.asarray(R, dtype=float)
        self.t = None if t is None else np.asarray(t, dtype=float)
        self.galvo = galvo if galvo is not None else GalvoSim()
        self.safe = False
        self.last_shot = None

    def sim_step(self, target_xyz, energy_j=3e-3, spot_um=300.0):
        """One sim step: project, drive one galvo tick, maybe abort."""
        theta_x, theta_y = project_angles(target_xyz, self.K, self.R, self.t)
        fx, fy = angles_to_field_mm(theta_x, theta_y)
        self.galvo.set_target(fx, fy)
        self.galvo.step()
        err = self.galvo.tracking_error_mm()
        if err > ABORT_ERROR_MM:
            self.safe = True
            return {"aborted": True, "error_mm": err, "safe": True}
        self.last_shot = evaluate_shot(energy_j, spot_um, fx, fy)
        return {
            "aborted": False,
            "error_mm": err,
            "safe": False,
            "theta": (theta_x, theta_y),
            "field_mm": (fx, fy),
            "shot": self.last_shot,
        }


def main():
    ap = argparse.ArgumentParser(description="galvo plugin sim-step harness")
    ap.add_argument("--sim-step", action="store_true")
    ap.add_argument("--tx", type=float, default=0.05)
    ap.add_argument("--ty", type=float, default=-0.03)
    ap.add_argument("--tz", type=float, default=3.0)
    args = ap.parse_args()
    plug = GalvoPlugin()
    if args.sim_step:
        res = plug.sim_step((args.tx, args.ty, args.tz))
        print(f"SIMULATED sim-step aborted={res['aborted']} error_mm={res['error_mm']:.3f}")
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
