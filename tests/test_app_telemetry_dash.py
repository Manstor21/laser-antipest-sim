"""PR2 RED — dashboard sim + JSONL replay (SIMULATED, sim-only 0EUR).

Spec: sdd/velutina-app-telemetry/spec #170 (app-dashboard-sim).
Design: #171 — app/dash_sim.py pull refresh(t), JSON+ASCII, NO-DATA.
Tasks: #173 Phase 2 (2.1 RED, 2.2 dash).

Sim-time t explícito float, SIMULATED=True, seed42 determinista.
"""

import hashlib
import json


def _h(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()


def test_dash_simulated_flag():
    from app import dash_sim
    from app import log_store

    assert dash_sim.SIMULATED is True
    assert log_store.SIMULATED is True


def test_render_full_cycle_t_1_0_1_2_1_5():
    from app.dash_sim import DashSim
    from telemetry.mqtt_sim import MqttSim

    bus = MqttSim(seed=42)
    dash = DashSim(bus)
    bus.publish("velutina/detecciones", t=1.0,
                payload={"conf": 0.97, "x9": [0.1] * 9}, qos=0)
    bus.publish("velutina/disparos", t=1.2,
                payload={"shot": {"e": 1}, "fsm_state": "FIRING"}, qos=0)
    bus.publish("velutina/estado", t=1.5,
                payload={"fsm_state": "TRACKING", "conf": 0.5}, qos=0)
    out = dash.refresh(t=1.5)
    assert set(out["json"].keys()) == {"detecciones", "disparos", "estado"}
    assert out["json"]["detecciones"]["conf"] == 0.97
    assert out["json"]["disparos"]["shot"] == {"e": 1}
    assert out["json"]["estado"]["fsm_state"] == "TRACKING"
    ascii_out = out["ascii"]
    assert "conf" in ascii_out
    assert "shot" in ascii_out
    assert "fsm_state" in ascii_out


def test_empty_disparos_shows_no_data():
    from app.dash_sim import DashSim
    from telemetry.mqtt_sim import MqttSim

    bus = MqttSim(seed=42)
    dash = DashSim(bus)
    bus.publish("velutina/detecciones", t=1.0,
                payload={"conf": 0.9}, qos=0)
    bus.publish("velutina/estado", t=1.5,
                payload={"fsm_state": "IDLE"}, qos=0)
    out = dash.refresh(t=1.5)
    assert out["json"]["disparos"] == "NO-DATA"
    assert "NO-DATA" in out["ascii"]
    # Las demás secciones renderizan normal.
    assert out["json"]["detecciones"] == {"conf": 0.9}


def test_jsonl_replay_equals_live_hash(tmp_path):
    from app.dash_sim import DashSim
    from app.log_store import LogStore
    from telemetry.mqtt_sim import MqttSim

    path = tmp_path / "events.jsonl"
    bus = MqttSim(seed=42)
    dash = DashSim(bus)
    store = LogStore(path)
    for i in range(100):
        t = 1.0 + i * 0.01
        ev = {"topic": "velutina/estado", "t": t,
              "payload": {"seq": i, "fsm_state": "TRACKING"}}
        bus.publish(ev["topic"], t=ev["t"], payload=ev["payload"], qos=0)
        store.append(ev)
    live = dash.refresh(t=2.0)
    live_hash = _h(live["json"])
    # Replay desde offset 0 en bus fresco → mismo hash final.
    bus2 = MqttSim(seed=42)
    dash2 = DashSim(bus2)
    store2 = LogStore(path)
    for ev in store2.replay(offset=0):
        bus2.publish(ev["topic"], t=ev["t"], payload=ev["payload"], qos=0)
    replayed = dash2.refresh(t=2.0)
    assert _h(replayed["json"]) == live_hash
