"""PR3 RED — link budget sim (SIMULATED, sim-only 0EUR).

Spec: sdd/velutina-app-telemetry/spec #170 (link-budget-sim).
Design: #171 — telemetry/link_budget.py fórmulas puras sobre
  LinkSim.dropped/queue+delay_for(); LoRa SF12/BW125 + WiFi MCS.
Tasks: #173 Phase 2 (2.6 RED link parte, 2.7 impl).

Sim-time t explícito float, SIMULATED=True, B1-4/sim-link solo lectura.
"""

import hashlib
import json


def _h(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()


def test_link_simulated_flag():
    from telemetry import link_budget

    assert link_budget.SIMULATED is True


def test_lora_sf12_range_golden_within_1pct():
    from sim.link.link_sim import LinkSim
    from telemetry import link_budget

    link = LinkSim(seed=42)
    out = link_budget.compute(link, radio="lora_sf12_bw125",
                              tx_dbm=14.0, sens_dbm=-137.0, t=0.0)
    # Golden Friis 868MHz: Lmax=151dB -> ~976.07 km (free-space ideal sim).
    assert out["range_km"] == __import__("pytest").approx(976.07, rel=0.01)
    assert out["warning"] is False
    assert out["goodput"] == __import__("pytest").approx(1.0)
    assert out["snr_db"] > 0
    assert "rssi_dbm" in out


def test_congested_link_degrades_and_warns():
    from sim.link.link_sim import LinkSim
    from telemetry import link_budget

    link = LinkSim(seed=42, capacity=16)
    clean = link_budget.compute(link, radio="lora_sf12_bw125",
                                tx_dbm=14.0, sens_dbm=-137.0, t=0.0)
    # Satura FIFO16: 16 + 4 -> dropped=4.
    for i in range(20):
        link.send(t=5.0, payload=bytes([i % 256] * 32))
    assert link.dropped == 4
    congested = link_budget.compute(link, radio="lora_sf12_bw125",
                                    tx_dbm=14.0, sens_dbm=-137.0, t=5.0)
    assert congested["warning"] is True
    assert congested["goodput"] < clean["goodput"]
    assert congested["effective_range_km"] < clean["range_km"]


def test_wifi_mcs_parametrized():
    from sim.link.link_sim import LinkSim
    from telemetry import link_budget

    link = LinkSim(seed=42)
    out = link_budget.compute(link, radio="wifi_mcs0",
                              tx_dbm=20.0, sens_dbm=-82.0, t=0.0)
    assert out["range_km"] == __import__("pytest").approx(1.2335, rel=0.01)
    assert out["warning"] is False


def test_link_compute_no_mutation():
    from sim.link.link_sim import LinkSim
    from telemetry import link_budget

    link = LinkSim(seed=42)
    for i in range(5):
        link.send(t=float(i), payload=bytes([1, 2, 3] * 8))
    pre = (_h([list(p["payload"]) for p in link._queue]), link.dropped)
    link_budget.compute(link, radio="lora_sf12_bw125",
                        tx_dbm=14.0, sens_dbm=-137.0, t=5.0)
    link_budget.compute(link, radio="wifi_mcs0",
                        tx_dbm=20.0, sens_dbm=-82.0, t=5.0)
    post = (_h([list(p["payload"]) for p in link._queue]), link.dropped)
    assert post == pre
