"""EKF->gate->FSM->galvo->f-theta wiring (SIMULATED, sim-only 0EUR, no HW).

Composition-only: consumes FusionPipeline outputs via fire_authorize(),
never mutates step()/step_with_ai(). Any abort/fault forces FSM SAFE.
Default is fail-closed: laser.enabled=False. Tests must pass explicit
laser_cfg with enabled=True. Production passes config.yaml where
laser.enabled=false unless explicitly enabled with safety review.

Realtime split (PR2): jetson_step(t) -> LinkSim -> mcu_step(t_fire).
cycle() kept as compat wrapper delegating to the 3 stages. Explicit
sim-time t everywhere, no sleep/time.time in core.
"""

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from firmware.galvo_shutter.galvo_sim import ABORT_ERROR_MM, GalvoSim
from firmware.galvo_shutter.shutter_sim import ShutterSim
from firmware.interlock_hw.hw_interlock import HwInterlock
from jetson.fusion.fusion_pipeline import fire_authorize
from jetson.fusion.hard_rules import r6_interlock
from jetson.laser.env_interlock import env_clear
from jetson.laser.fire_controller import FireController
from jetson.laser.flags import laser_enabled
from sim.gazebo.f_theta_optics import evaluate_shot
from sim.gazebo.galvo_plugin import angles_to_field_mm, project_angles
from sim.link.link_sim import LinkSim

SIMULATED = True
SETTLE_STEPS = 40
LINK_AUTH_BYTES = 32


class FirePipeline:
    """Full fire loop: gate -> FSM -> galvo -> f-theta (SIMULATED)."""

    def __init__(self, K=None, R=None, t=None, galvo=None,
                 shutter=None, controller=None, laser_cfg=None,
                 link=None, hw=None, mcu=None, explicit_config=False):
        self.K = K
        self.R = R
        self.t = t
        self.galvo = galvo if galvo is not None else GalvoSim()
        self.shutter = shutter if shutter is not None else ShutterSim()
        self.controller = controller if controller is not None else FireController()
        if laser_cfg is None:
            # Fail-closed default: laser disabled unless explicitly configured.
            # Tests that need the laser must pass laser_cfg with enabled=True.
            laser_cfg = {"laser": {"enabled": False, "simulated": True}}
        self.laser_cfg = laser_cfg
        # Fail-closed validation: if laser.enabled=True but no explicit config
        # was provided (e.g. from config.yaml), log WARNING and treat as disabled.
        laser_on = laser_cfg.get("laser", {}).get("enabled", False)
        if laser_on and not explicit_config:
            logging.warning(
                "FirePipeline: laser.enabled=True but no explicit config provided. "
                "Fail-closed: treating as disabled. Pass explicit_config=True or "
                "load from config.yaml to enable."
            )
            self.laser_cfg = {"laser": {"enabled": False, "simulated": True}}
        # DI defaults: link=LinkSim UART115200/FIFO16/seed42, hw=HwInterlock
        # pass-through when inputs clear, mcu=None (optional VirtualMcu tick).
        # Open Q: SPI vs UART default; FIFO 16 vs 32 under EKF 10Hz burst.
        self.link = link if link is not None else LinkSim()
        self.hw = hw if hw is not None else HwInterlock
        self.mcu = mcu

    def jetson_step(self, fusion_out, target_xyz, t=0.0,
                    human_min_m=5.0, track_mm=10.0,
                    temp_c=22.0, rh_pct=55.0, material="soil"):
        """Jetson stage: gating + field projection, publishes authorize msg."""
        t = float(t)
        laser_on = laser_enabled(self.laser_cfg)
        interlock_clear = bool(r6_interlock(human_min_m, track_mm))
        env_ok = bool(env_clear(temp_c, rh_pct, material))
        try:
            pre_gate = bool(
                isinstance(fusion_out, dict)
                and bool(fusion_out.get("promoted"))
                and fusion_out.get("ai_vote") is True
                and isinstance(fusion_out.get("gates"), dict)
                and all(bool(v) for v in fusion_out["gates"].values())
                and interlock_clear and env_ok and laser_on
            )
        except (AttributeError, TypeError, ValueError):
            pre_gate = False
        # t_meas propagates from fusion when available, else authorize time.
        try:
            t_meas = float(fusion_out.get("t_meas", t)) if isinstance(fusion_out, dict) else float(t)
        except (TypeError, ValueError):
            t_meas = float(t)
        theta_x, theta_y = project_angles(target_xyz, self.K, self.R, self.t)
        fx, fy = angles_to_field_mm(theta_x, theta_y)
        # Transport 32B authorize frame over sim link (deterministic delay).
        payload = (b"\x01" if pre_gate else b"\x00") * LINK_AUTH_BYTES
        sent = self.link.send(t, payload)
        t_recv = float(sent["t_recv"])
        # Drain due packets at t_recv to keep FIFO consistent (never blocks).
        try:
            self.link.recv(t_recv)
        except (AttributeError, TypeError, ValueError):
            pass
        return {
            "authorized": bool(pre_gate),
            "t_authorize": float(t),
            "t_meas": float(t_meas),
            "t_recv": float(t_recv),
            "field_mm": (float(fx), float(fy)),
            "fusion_out": fusion_out,
            "interlock_clear": bool(interlock_clear),
            "env_ok": bool(env_ok),
            "laser_on": bool(laser_on),
        }

    def mcu_step(self, msg, t_fire=None, hw_inputs=None,
                 width_us=3.0, energy_j=3e-3, spot_um=300.0,
                 field_override_mm=None):
        """MCU stage: shutter+HW veto+FSM mirror+galvo, explicit t_fire."""
        if not isinstance(msg, dict):
            raise ValueError("msg must be jetson_step dict")
        t_authorize = float(msg.get("t_authorize", 0.0))
        t_recv = float(msg.get("t_recv", t_authorize))
        t_fire = float(t_recv if t_fire is None else t_fire)
        fusion_out = msg.get("fusion_out")
        field_mm = msg.get("field_mm", (0.0, 0.0))
        if field_override_mm is not None:
            fx, fy = float(field_override_mm[0]), float(field_override_mm[1])
        else:
            fx, fy = float(field_mm[0]), float(field_mm[1])
        # Optional VirtualMcu tick hook (cooperative, never blocks fire path).
        if self.mcu is not None:
            try:
                self.mcu.step(t_fire)
            except (AttributeError, TypeError, ValueError):
                pass
        # Shutter feed/command driven by jetson authorize flag.
        jetson_auth = bool(msg.get("authorized"))
        self.shutter.feed_watchdog(t_fire)
        self.shutter.command(open=bool(jetson_auth), t=t_fire)
        shutter_ready = bool(self.shutter.evaluate(t_fire))
        interlock_clear = bool(msg.get("interlock_clear", False))
        env_ok = bool(msg.get("env_ok", True))
        laser_on = bool(msg.get("laser_on", True))
        authorized = bool(
            fire_authorize(fusion_out, interlock_clear, shutter_ready)
            and env_ok and laser_on and jetson_auth
        )
        # HW veto overrides SW authorize (<100us by construction, pure eval).
        vetoed = False
        if hw_inputs is not None:
            try:
                hw_ok = bool(self.hw.eval(hw_inputs, t_fire))
            except (AttributeError, TypeError, ValueError):
                hw_ok = False
            if not hw_ok:
                vetoed = True
        if vetoed:
            self.controller.fault(t=t_fire, reason="hw-interlock-veto")
            return {
                "authorized": bool(authorized),
                "fsm_state": self.controller.state,
                "aborted": True,
                "safe": True,
                "vetoed": True,
                "error_mm": float(self.galvo.tracking_error_mm()),
                "shot": None,
                "t_authorize": float(t_authorize),
                "t_fire": float(t_fire),
                "t_meas": float(msg.get("t_meas", t_authorize)),
            }
        if not authorized:
            return {
                "authorized": False,
                "fsm_state": self.controller.state,
                "aborted": False,
                "safe": self.controller.state == "SAFE",
                "vetoed": False,
                "error_mm": float(self.galvo.tracking_error_mm()),
                "shot": None,
                "t_authorize": float(t_authorize),
                "t_fire": float(t_fire),
                "t_meas": float(msg.get("t_meas", t_authorize)),
            }
        self.controller.try_arm(authorized=True, t=t_fire)
        self.galvo.set_target(fx, fy)
        self.galvo.step()
        err = float(self.galvo.tracking_error_mm())
        if err > float(ABORT_ERROR_MM):
            self.controller.fault(t=t_fire, reason="tracking-abort")
            return {
                "authorized": True,
                "fsm_state": self.controller.state,
                "aborted": True,
                "safe": True,
                "vetoed": False,
                "error_mm": err,
                "shot": None,
                "t_authorize": float(t_authorize),
                "t_fire": float(t_fire),
                "t_meas": float(msg.get("t_meas", t_authorize)),
                "settle_ts": self.galvo.settle_timestamp(),
                "abort_ts": self.galvo.abort_timestamp(),
            }
        for _ in range(SETTLE_STEPS - 1):
            self.galvo.step()
        err = float(self.galvo.tracking_error_mm())
        if err > float(ABORT_ERROR_MM):
            self.controller.fault(t=t_fire, reason="tracking-abort")
            return {
                "authorized": True,
                "fsm_state": self.controller.state,
                "aborted": True,
                "safe": True,
                "vetoed": False,
                "error_mm": err,
                "shot": None,
                "t_authorize": float(t_authorize),
                "t_fire": float(t_fire),
                "t_meas": float(msg.get("t_meas", t_authorize)),
                "settle_ts": self.galvo.settle_timestamp(),
                "abort_ts": self.galvo.abort_timestamp(),
            }
        settle_ok = err <= float(ABORT_ERROR_MM)
        self.controller.start_firing(settle_ok=settle_ok, t=t_fire)
        if self.controller.state != "FIRING":
            return {
                "authorized": True,
                "fsm_state": self.controller.state,
                "aborted": False,
                "safe": self.controller.state == "SAFE",
                "vetoed": False,
                "error_mm": err,
                "shot": None,
                "t_authorize": float(t_authorize),
                "t_fire": float(t_fire),
                "t_meas": float(msg.get("t_meas", t_authorize)),
                "settle_ts": self.galvo.settle_timestamp(),
                "abort_ts": self.galvo.abort_timestamp(),
            }
        self.controller.request_pulse(t=t_fire, width_us=width_us)
        shot = evaluate_shot(energy_j, spot_um, fx, fy)
        return {
            "authorized": True,
            "fsm_state": self.controller.state,
            "aborted": False,
            "safe": False,
            "vetoed": False,
            "error_mm": err,
            "shot": shot,
            "t_authorize": float(t_authorize),
            "t_fire": float(t_fire),
            "t_meas": float(msg.get("t_meas", t_authorize)),
            "settle_ts": self.galvo.settle_timestamp(),
            "abort_ts": self.galvo.abort_timestamp(),
        }

    def cycle(self, fusion_out, target_xyz, human_min_m=5.0, track_mm=10.0,
              t=0.0, temp_c=22.0, rh_pct=55.0, material="soil",
              energy_j=3e-3, spot_um=300.0, width_us=3.0,
              field_override_mm=None):
        """Compat wrapper: jetson_step -> link -> mcu_step (no HW veto)."""
        msg = self.jetson_step(
            fusion_out, target_xyz, t=t,
            human_min_m=human_min_m, track_mm=track_mm,
            temp_c=temp_c, rh_pct=rh_pct, material=material,
        )
        return self.mcu_step(
            msg, t_fire=msg["t_recv"],
            hw_inputs=None, width_us=width_us,
            energy_j=energy_j, spot_um=spot_um,
            field_override_mm=field_override_mm,
        )


def main():
    ap = argparse.ArgumentParser(description="fire pipeline sim-step harness")
    ap.add_argument("--sim-step", action="store_true")
    args = ap.parse_args()
    if args.sim_step:
        # Test harness: explicitly pass laser_cfg with enabled=True and explicit_config=True
        pipe = FirePipeline(
            laser_cfg={"laser": {"enabled": True, "simulated": True}},
            explicit_config=True,
        )
        res = pipe.cycle(
            fusion_out={"promoted": True, "ai_vote": True,
                        "gates": {"R1": True, "R2": True, "R3": True, "R4": True}},
            target_xyz=(0.05, -0.03, 3.0),
            t=1.0,
        )
        print(f"SIMULATED sim-step authorized={res['authorized']} "
              f"fsm={res['fsm_state']} aborted={res['aborted']} "
              f"error_mm={res['error_mm']:.3f}")
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
