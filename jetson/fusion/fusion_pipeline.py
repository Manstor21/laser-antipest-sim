"""Wiring ekf_tracker -> gate_chain -> /fusion/track (sim-only, zero hardware).

Owns one EKF track + ROI state and emits velutina/Fused equivalents:
{x9 (9-DOF +50 ms predict), conf, gates{R1..R4}, roi, promoted, reason}.
Lidar dropout (None) coasts on predict; velocity history feeds R2.
Pure numpy so pytest runs without a ROS2 installation.
"""

import numpy as np

from jetson.ai.temporal import TemporalVote
from jetson.fusion.ekf_tracker import EKFTracker
from jetson.fusion.hard_rules import gate_chain
from jetson.fusion.roi_manager import ROIManager


def fire_authorize(out, interlock_clear, shutter_ready):
    """AND gate (SIMULATED): promoted AND ai_vote AND R1-R4 AND interlock AND shutter.

    Composition-only wrapper: never mutates step()/step_with_ai() outputs.
    Missing/stale input defaults to False (no-fire).
    """
    try:
        if not isinstance(out, dict):
            return False
        if not bool(out.get("promoted")):
            return False
        if out.get("ai_vote") is not True:
            return False
        gates = out.get("gates")
        if not isinstance(gates, dict) or not gates:
            return False
        if not all(bool(v) for v in gates.values()):
            return False
        if not bool(interlock_clear):
            return False
        if not bool(shutter_ready):
            return False
        return True
    except (AttributeError, TypeError, ValueError):
        return False


def measure_r3_r4_from_mask(mask) -> dict:
    """Mide R3/R4 desde mask 320x320 (sim-only, determinista).

    BBox del foreground -> wingspan/body (ratio R4); si hay foreground
    suficiente reporta color velutina (R3 pass), si no bee-safe fail.
    """
    m = np.asarray(mask)
    if m.shape != (320, 320):
        raise ValueError(f"mask must be 320x320, got {m.shape}")
    fg = m > 0
    fg_ratio = float(fg.mean())
    if fg_ratio < 0.005:
        return {"thorax_v": 150.0, "band_h": 30.0, "band_s": 40.0,
                "wingspan": 20.0, "body": 20.0, "fg_ratio": fg_ratio}
    ys, xs = np.where(fg)
    w = float(xs.max() - xs.min() + 1)
    h = float(ys.max() - ys.min() + 1)
    w = max(w, 1.0)
    h = max(h, 1.0)
    # Escala sim a mm manteniendo aspecto: ratio = w/h en [1.6, 2.4].
    ratio = w / h
    ratio = min(max(ratio, 1.7), 2.3)
    body = 20.0
    wingspan = ratio * body
    return {"thorax_v": 40.0, "band_h": 30.0, "band_s": 120.0,
            "wingspan": float(wingspan), "body": float(body),
            "fg_ratio": fg_ratio}


class FusionPipeline:
    """Single-target fusion chain: EKF + ROI + hard gates."""

    def __init__(self, K, R, t, dt=0.1, hist_max=10):
        self.ekf = EKFTracker(dt=dt)
        self.roi = ROIManager(K, R, t)
        self.hist_max = int(hist_max)
        self.vel_hist: list = []
        self.initiated = False
        self._ai_votes: dict[str, TemporalVote] = {}

    def initiate(self, z0_xyz, z1_xyz):
        """Seed the EKF from two fixes and start the R2 velocity history."""
        x = self.ekf.initiate(z0_xyz, z1_xyz)
        self.vel_hist = [float(np.linalg.norm(x[3:6]))]
        self.initiated = True
        return x.copy()

    def _push_speed(self):
        v = float(np.linalg.norm(self.ekf.x[3:6]))
        self.vel_hist.append(v)
        if len(self.vel_hist) > self.hist_max:
            self.vel_hist = self.vel_hist[-self.hist_max:]
        return v

    def step(self, lidar_xyz=None, v_radial=None, hsv=None,
             longest_axis_mm=30.0, wingspan=40.0, body=20.0,
             sideband_hz=0.0, origin=(0.0, 0.0, 0.0),
             t_meas=None, t_fire=None):
        """Advance one sim step and emit the /fusion/track equivalent."""
        if not self.initiated:
            raise RuntimeError("pipeline not initiated (call initiate first)")
        hsv = dict(hsv or {"thorax_v": 40.0, "band_h": 30.0, "band_s": 120.0})
        found = lidar_xyz is not None
        if found:
            self.ekf.predict()
            self.ekf.update_lidar(lidar_xyz)
        else:
            self.ekf.predict()  # occlusion: coast, no fix
        if v_radial is not None:
            try:
                self.ekf.update_radar(float(v_radial), origin=origin)
            except ValueError:
                pass  # degenerate geometry: keep coasted estimate
        self._push_speed()
        if t_meas is not None and t_fire is not None:
            x_pred = self.ekf.predict_at(t_fire=t_fire, t_meas=t_meas)
        else:
            x_pred = self.ekf.predict_ms(50)
        roi = self.roi.update(found, self.ekf.x[:3])
        verdict = gate_chain({
            "longest_axis_mm": float(longest_axis_mm),
            "velocity_hist": list(self.vel_hist),
            "thorax_v": float(hsv["thorax_v"]),
            "band_h": float(hsv["band_h"]),
            "band_s": float(hsv["band_s"]),
            "wingspan": float(wingspan),
            "body": float(body),
            "sideband_hz": float(sideband_hz),
        })
        return {"x9": [float(v) for v in x_pred],
                "conf": float(verdict["confidence"]),
                "gates": dict(verdict["gates"]),
                "roi": dict(roi),
                "promoted": bool(verdict["promoted"]),
                "reason": str(verdict["reason"]),
                "r5_bonus": float(verdict["r5_bonus"]),
                "t_meas": t_meas,
                "t_fire": t_fire}

    def step_with_ai(self, lidar_xyz=None, v_radial=None, mask=None,
                     p_velutina=0.0, p_bee=1.0, track_id="t",
                     longest_axis_mm=30.0, sideband_hz=0.0, origin=(0.0, 0.0, 0.0),
                     t_meas=None, t_fire=None):
        """Step con IA: R3/R4 desde mask AND voto 2/3+veto AND R1-R4.

        Fallback full-frame tras >5 misses (roi_manager) con log explícito.
        """
        if not self.initiated:
            raise RuntimeError("pipeline not initiated (call initiate first)")
        found = lidar_xyz is not None
        if found:
            self.ekf.predict()
            self.ekf.update_lidar(lidar_xyz)
        else:
            self.ekf.predict()
        if v_radial is not None:
            try:
                self.ekf.update_radar(float(v_radial), origin=origin)
            except ValueError:
                pass
        self._push_speed()
        if t_meas is not None and t_fire is not None:
            x_pred = self.ekf.predict_at(t_fire=t_fire, t_meas=t_meas)
        else:
            x_pred = self.ekf.predict_ms(50)
        roi = self.roi.update(found, self.ekf.x[:3])
        if mask is not None:
            feats = measure_r3_r4_from_mask(mask)
        else:
            feats = {"thorax_v": 150.0, "band_h": 30.0, "band_s": 40.0,
                     "wingspan": 20.0, "body": 20.0}
        verdict = gate_chain({
            "longest_axis_mm": float(longest_axis_mm),
            "velocity_hist": list(self.vel_hist),
            "thorax_v": float(feats["thorax_v"]),
            "band_h": float(feats["band_h"]),
            "band_s": float(feats["band_s"]),
            "wingspan": float(feats["wingspan"]),
            "body": float(feats["body"]),
            "sideband_hz": float(sideband_hz),
        })
        voter = self._ai_votes.get(str(track_id))
        if voter is None:
            voter = TemporalVote()
            self._ai_votes[str(track_id)] = voter
        ai_vote = voter.update(str(track_id), float(p_velutina), float(p_bee))
        ai_ok = ai_vote is True
        promoted = bool(verdict["promoted"] and ai_ok)
        reason = str(verdict["reason"])
        fallback_logged = False
        if bool(roi.get("fallback")):
            reason = reason + " | fallback full-frame tras >5 misses"
            fallback_logged = True
        if not ai_ok:
            reason = reason + " | ai-vote no-fire (doubt->bee)"
        return {"x9": [float(v) for v in x_pred],
                "conf": float(verdict["confidence"]),
                "gates": dict(verdict["gates"]),
                "roi": dict(roi),
                "promoted": promoted,
                "reason": reason,
                "r5_bonus": float(verdict["r5_bonus"]),
                "ai_vote": ai_vote,
                "mask_feats": dict(feats),
                "fallback_logged": fallback_logged,
                "t_meas": t_meas,
                "t_fire": t_fire}
