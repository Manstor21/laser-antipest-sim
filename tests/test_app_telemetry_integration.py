"""PR3 RED — integración fusión→bus→dash→JSONL + no-mutación (SIMULATED).

Spec #170 (mqtt-sim-bus no-mutación + dashboard replay) + Tasks #173
Phase 3 (3.1 replay 100 eventos seed42 hash live==replay, 3.2 green +
SIMULATED + B1-4 sin diff) y Phase 4 (4.1 docstrings SIMULATED).

Sim-time t explícito, seed42, 0EUR sim-only, B1-4 solo lectura.
"""

import hashlib
import json


def _h(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()


def test_integration_cycle_replay_hash_equal(tmp_path):
    from app.dash_sim import DashSim
    from app.log_store import LogStore
    from telemetry.mqtt_sim import MqttSim

    path = tmp_path / "cycle.jsonl"
    bus = MqttSim(seed=42)
    dash = DashSim(bus)
    store = LogStore(path)
    for i in range(100):
        t = 1.0 + i * 0.01
        topic = ("velutina/detecciones" if i % 3 == 0
                 else "velutina/disparos" if i % 3 == 1
                 else "velutina/estado")
        payload = {"seq": i, "conf": 0.90 + (i % 10) * 0.005,
                   "fsm_state": "TRACKING"}
        bus.publish(topic, t=t, payload=payload, qos=0)
        store.append({"topic": topic, "t": t, "payload": payload})
    live_hash = _h(dash.refresh(t=2.0)["json"])
    bus2 = MqttSim(seed=42)
    dash2 = DashSim(bus2)
    for ev in LogStore(path).replay(offset=0):
        bus2.publish(ev["topic"], t=ev["t"], payload=ev["payload"], qos=0)
    assert _h(dash2.refresh(t=2.0)["json"]) == live_hash


def test_integration_no_mutation_and_yaml_intact():
    from app.config_sim import real_checksums
    from telemetry.mqtt_sim import (MqttSim, snapshot_fire, snapshot_fusion,
                                    snapshot_link)

    before = real_checksums()
    fusion_d = {"x9": [0.5] * 9, "conf": 0.97, "gates": {"R1": True}}
    fire_d = {"fsm_state": "FIRING", "shot": {"e": 1}}
    link_d = {"dropped": 0, "queue": [1, 2]}
    pre = (_h(fusion_d), _h(fire_d), _h(link_d))
    bus = MqttSim(seed=42)
    bus.publish("velutina/detecciones", t=10.0,
                payload=snapshot_fusion(fusion_d))
    bus.publish("velutina/disparos", t=10.0, payload=snapshot_fire(fire_d))
    bus.publish("velutina/estado", t=10.0, payload=snapshot_link(link_d))
    assert (_h(fusion_d), _h(fire_d), _h(link_d)) == pre
    assert real_checksums() == before


def test_all_new_modules_simulated_and_docstrings():
    import app.config_sim as ac
    import app.dash_sim as ad
    import app.log_store as al
    from telemetry import link_budget, mqtt_sim, ota_sim

    for mod in (mqtt_sim, link_budget, ota_sim, ad, ac, al):
        assert mod.SIMULATED is True
        doc = (mod.__doc__ or "").upper()
        assert "SIMULATED" in doc
