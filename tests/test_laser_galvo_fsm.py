"""RED 1.2 — FSM fault->SAFE <10ms sim-only (must fail before impl, PR1)."""
import pytest

from jetson.laser.fire_controller import DutyMeter, FireController, pulse_energy_j


def _armed_firing(t0=100.0):
    fc = FireController()
    assert fc.state == "IDLE"
    fc.try_arm(authorized=True, t=t0)
    assert fc.state == "ARMED"
    fc.start_firing(settle_ok=True, t=t0 + 0.001)
    assert fc.state == "FIRING"
    return fc


def test_idle_requires_authorize():
    fc = FireController()
    assert fc.state == "IDLE"
    fc.try_arm(authorized=False, t=0.0)
    assert fc.state == "IDLE"
    assert fc.request_pulse(t=0.001, width_us=3.0)[0] is False


def test_arm_then_fire_window():
    fc = _armed_firing()
    ok, energy_j = fc.request_pulse(t=100.002, width_us=3.0)
    assert ok is True
    assert energy_j == pytest.approx(0.003, rel=1e-6)


def test_fault_during_firing_to_safe_lt10ms():
    fc = _armed_firing()
    t_fault = 100.005
    fc.fault(t=t_fault, reason="watchdog")
    assert fc.state == "SAFE"
    assert (fc.safe_enter_t - t_fault) < 0.010
    assert fc.emission_on is False
    ok, _ = fc.request_pulse(t=t_fault + 0.001, width_us=3.0)
    assert ok is False


@pytest.mark.parametrize("start", ["IDLE", "ARMED", "FIRING"])
def test_any_fault_forces_safe(start):
    fc = FireController()
    if start in ("ARMED", "FIRING"):
        fc.try_arm(authorized=True, t=1.0)
    if start == "FIRING":
        fc.start_firing(settle_ok=True, t=1.001)
    assert fc.state == start
    fc.fault(t=2.0, reason="interlock")
    assert fc.state == "SAFE"
    assert fc.emission_on is False


def test_energy_accounting_3us_3mJ():
    assert pulse_energy_j(1000.0, 3.0) == pytest.approx(0.003, rel=1e-9)
    assert pulse_energy_j(1000.0, 1.0) == pytest.approx(0.001, rel=1e-9)
    assert pulse_energy_j(1000.0, 5.0) == pytest.approx(0.005, rel=1e-9)


def test_duty_meter_blocks_creep():
    dm = DutyMeter(limit=0.001, window_s=1.0)
    t = 0.0
    # 120Hz x 10us pulses for 1s -> duty 0.12% > 0.1%: must block near end.
    allowed = [dm.request(t=t + i / 120.0, width_us=10.0) for i in range(120)]
    assert False in allowed  # must block once >0.1%
    assert allowed[99] is True  # 100 x 10us = 1000us = limit still ok
    # latched: next request still denied until window decays
    assert dm.request(t=0.995, width_us=10.0) is False


def test_burst_cooldown_enforced():
    fc = _armed_firing(t0=50.0)
    t = 50.002
    for i in range(5):
        ok, _ = fc.request_pulse(t=t + i * 0.001, width_us=3.0)
        assert ok is True
    ok, _ = fc.request_pulse(t=t + 0.006, width_us=3.0)
    assert ok is False  # burst limit 5
    # after 2s cooldown, burst resets but FSM stays FIRING
    ok, _ = fc.request_pulse(t=t + 2.1, width_us=3.0)
    assert ok is True
