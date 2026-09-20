"""RED test 2.4b — radar sim velocity + R5 (must fail before impl)."""
from sim.ros2.radar_sim import RadarSim


def test_hover_vs_transit():
    radar = RadarSim(seed=5)
    hover = [radar.observe(0.1, sideband_hz=0.0) for _ in range(3)]
    assert all(o["label"] == "hover" for o in hover)
    transit = [radar.observe(8.0, sideband_hz=130.0) for _ in range(3)]
    assert all(o["label"] == "transit" for o in transit)
    err = abs(sum(o["v_meas"] for o in transit) / 3 - 8.0)
    assert err <= 0.3


def test_sideband_advisory_ordering():
    radar = RadarSim(seed=5)
    with_band = radar.observe(6.0, sideband_hz=130.0)
    without = radar.observe(6.0, sideband_hz=0.0)
    assert with_band["priority"] > without["priority"]
    assert without["rejected_by_r5"] is False  # R5 never gates alone
