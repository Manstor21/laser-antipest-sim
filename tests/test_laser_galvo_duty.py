"""RED 1.3 — duty/energy/burst budget (must fail before impl, PR2 sim-only)."""
import pytest

from jetson.laser.fire_controller import DutyMeter, FireController, pulse_energy_j


def test_duty_creep_120hz_blocks_and_latches():
    dm = DutyMeter(limit=0.001, window_s=1.0)
    # 120Hz x 10us = 0.12% in 1s -> must block once >0.1%.
    allowed = [dm.request(t=i / 120.0, width_us=10.0) for i in range(120)]
    assert False in allowed
    assert allowed[99] is True  # 100 x 10us = 1000us = limit still ok
    assert dm.request(t=0.995, width_us=10.0) is False  # latched until decay
    # after window decays, requests allowed again
    assert dm.request(t=2.000, width_us=10.0) is True


def test_energy_3us_is_3mJ():
    assert pulse_energy_j(1000.0, 3.0) == pytest.approx(0.003, rel=1e-9)
    assert pulse_energy_j(1000.0, 1.0) == pytest.approx(0.001, rel=1e-9)
    assert pulse_energy_j(1000.0, 5.0) == pytest.approx(0.005, rel=1e-9)


def test_burst_window_denied_latched_until_cooldown():
    fc = FireController()
    fc.try_arm(authorized=True, t=10.0)
    fc.start_firing(settle_ok=True, t=10.001)
    t = 10.002
    for i in range(5):
        ok, energy = fc.request_pulse(t=t + i * 0.001, width_us=3.0)
        assert ok is True
        assert energy == pytest.approx(0.003, rel=1e-6)
    ok, _ = fc.request_pulse(t=t + 0.006, width_us=3.0)
    assert ok is False  # burst-limit 5, denied+latched
    ok, _ = fc.request_pulse(t=t + 0.500, width_us=3.0)
    assert ok is False  # still latched before 2s cooldown
    ok, _ = fc.request_pulse(t=t + 2.100, width_us=3.0)
    assert ok is True  # cooldown elapsed -> allowed
