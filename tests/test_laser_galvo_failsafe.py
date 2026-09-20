"""RED 1.4 — failsafe shutter NC + human inhibit (must fail before impl, PR2 sim-only)."""
import pytest

from firmware.galvo_shutter.shutter_sim import (
    ACCESSIBLE_LIMIT_J,
    OD_CLOSED,
    ShutterSim,
    accessible_energy_j,
)
from jetson.fusion.fusion_pipeline import fire_authorize
from jetson.fusion.hard_rules import r6_interlock


def _auth_out():
    return {
        "promoted": True,
        "ai_vote": True,
        "gates": {"R1": True, "R2": True, "R3": True, "R4": True},
    }


def test_closed_attenuation_below_1p8uJ():
    for energy_mj in (1.0, 3.0, 5.0):
        acc = accessible_energy_j(energy_mj * 1e-3, shutter_open=False)
        assert acc < ACCESSIBLE_LIMIT_J
    assert OD_CLOSED >= 3.5
    # open shutter passes full energy (sim Class 4)
    assert accessible_energy_j(3e-3, shutter_open=True) == pytest.approx(3e-3)


def test_stuck_open_mismatch_forces_safe_lt10ms():
    sh = ShutterSim()
    sh.command(close=True, t=0.0)
    sh.feed_watchdog(t=0.0)
    # inject stuck-open: feedback reports open while commanded closed
    sh.inject_stuck_open(True)
    fault_t = 0.005
    assert sh.evaluate(t=fault_t) is False  # not ready -> mismatch
    assert sh.mismatch is True
    # controller view: mismatch forces SAFE <10ms, emission stops
    from jetson.laser.fire_controller import FireController

    fc = FireController()
    fc.try_arm(authorized=True, t=0.0)
    fc.start_firing(settle_ok=True, t=0.001)
    fc.fault(t=fault_t, reason="shutter-mismatch")
    assert fc.state == "SAFE"
    assert (fc.safe_enter_t - fault_t) < 0.010
    assert fc.emission_on is False


def test_watchdog_loss_forces_safe():
    sh = ShutterSim()
    sh.command(close=False, t=0.0)
    sh.feed_watchdog(t=0.0)
    # no feed for >10ms -> watchdog trips, shutter not ready
    assert sh.evaluate(t=0.005) is True
    assert sh.evaluate(t=0.020) is False
    assert sh.watchdog_ok(t=0.020) is False


def test_human_1p5m_inhibits_fire():
    assert r6_interlock(1.5, 10.0) is False
    out = _auth_out()
    assert fire_authorize(out, interlock_clear=r6_interlock(1.5, 10.0), shutter_ready=True) is False
