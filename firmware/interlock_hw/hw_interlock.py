"""HW interlock pure combinational (SIMULATED, sim-only 0EUR, no HW).

DI default: polled every 10us (POLL_S), cut <100us by construction.
No FSM/CPU/shutter imports — stays import-free from SW.
"""
from dataclasses import dataclass

SIMULATED = True
POLL_S = 10e-6


@dataclass
class HwInputs:
    enable_sw: bool
    r6_hw: bool
    shutter_fb_closed: bool
    wdog_hw_ok: bool


class HwInterlock:
    """Pure eval: fire iff SW enable and no HW veto."""

    @staticmethod
    def eval(inp, t):
        _ = float(t)
        return bool(inp.enable_sw and not inp.r6_hw
                    and inp.shutter_fb_closed and inp.wdog_hw_ok)
