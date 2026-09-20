"""RED 1.5 — optics f-theta + galvo settle + budget/cooldown (must fail before impl, PR2)."""
import pytest

from firmware.galvo_shutter.galvo_sim import GalvoSim
from sim.gazebo.f_theta_optics import SPOT_MAX_UM, SPOT_MIN_UM, evaluate_shot, fluence_Jcm2


def test_fluence_range_0p5_to_15_Jcm2():
    # 1mJ @ 500um -> ~0.51 J/cm2 (low end); 5mJ @ 200um -> ~15.9 J/cm2 (high end)
    assert fluence_Jcm2(1e-3, 500.0) == pytest.approx(0.51, rel=0.05)
    assert fluence_Jcm2(5e-3, 200.0) == pytest.approx(15.9, rel=0.05)
    assert SPOT_MIN_UM == 200.0
    assert SPOT_MAX_UM == 500.0


def test_spot_out_of_range_rejected():
    with pytest.raises(ValueError):
        fluence_Jcm2(3e-3, 100.0)
    with pytest.raises(ValueError):
        fluence_Jcm2(3e-3, 600.0)


def test_galvo_settle_then_fire_window():
    g = GalvoSim()
    g.set_target(5.0, -3.0)
    # integrate 2ms at 50us steps; 2nd-order XY2-100 must settle <500us band
    for _ in range(40):
        g.step()
    assert g.settled(band_mm=0.5) is True
    assert g.settle_time_s() < 500e-6
    assert g.tracking_error_mm() <= 10.0


def test_galvo_tracking_abort_15mm():
    g = GalvoSim()
    g.set_target(0.0, 0.0)
    for _ in range(10):
        g.step()
    # fast target jump emulating 5-8m/s tracking loss -> 15mm error
    g.set_target(15.0, 0.0)
    g.step()
    assert g.tracking_error_mm() > 10.0
    assert g.abort_required() is True


def test_optics_evaluate_logs_fluence_and_cooldown_budget():
    from jetson.laser.fire_controller import FireController

    shot = evaluate_shot(energy_j=3e-3, spot_um=300.0, x_mm=5.0, y_mm=-3.0)
    assert shot["fluence_Jcm2"] == pytest.approx(fluence_Jcm2(3e-3, 300.0))
    assert shot["spot_um"] == 300.0
    # dry burst-limit: 5 pulses ok, 6th denied until 2s cooldown
    fc = FireController()
    fc.try_arm(authorized=True, t=20.0)
    fc.start_firing(settle_ok=True, t=20.001)
    t = 20.002
    for i in range(5):
        assert fc.request_pulse(t=t + i * 0.001, width_us=3.0)[0] is True
    assert fc.request_pulse(t=t + 0.006, width_us=3.0)[0] is False
    assert fc.request_pulse(t=t + 2.100, width_us=3.0)[0] is True
