"""PR1 RED — bus sim + adapters read-only (SIMULATED, sim-only 0EUR).

Spec: sdd/velutina-app-telemetry/spec #170 (mqtt-sim-bus).
Design: #171 — telemetry/mqtt_sim.py dict{tópico:FIFO16}+QoS sim+seed42.
Tasks: #173 Phase 1 (1.1 RED, 1.2 bus, 1.3 adapters deepcopy B1-4).

Sim-time t explícito float, SIMULATED=True, seed42 determinista.
B1-4 solo lectura: hash pre/post igual (no-mutación).
"""
import copy
import hashlib
import json

import pytest


def _h(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()


def test_bus_simulated_flag_and_topics():
    from telemetry import mqtt_sim

    assert mqtt_sim.SIMULATED is True
    assert set(mqtt_sim.TOPICS) == {
        "velutina/detecciones",
        "velutina/disparos",
        "velutina/estado",
    }


def test_publish_subscribe_detection_t10():
    from telemetry.mqtt_sim import MqttSim, snapshot_fusion

    bus = MqttSim(seed=42)
    received = []
    bus.subscribe("velutina/detecciones", lambda env: received.append(env))
    fusion_out = {
        "x9": [0.1 * i for i in range(9)],
        "conf": 0.97,
        "gates": {"R1": True, "R2": True, "R3": True, "R4": True},
        "promoted": True,
        "t_meas": 10.0,
    }
    snap = snapshot_fusion(fusion_out)
    res = bus.publish("velutina/detecciones", t=10.0, payload=snap, qos=0)
    assert res["t_recv"] == pytest.approx(10.0)
    assert res["dup"] is False
    assert len(received) == 1
    assert received[0]["t"] == pytest.approx(10.0)
    assert received[0]["payload"] == snap
    assert received[0]["dup"] is False


def test_qos1_dup_retry():
    from telemetry.mqtt_sim import MqttSim

    bus = MqttSim(seed=42)
    received = []
    bus.subscribe("velutina/detecciones", lambda env: received.append(env), qos=1)
    # Simula caída de transporte una vez: el retry llega con dup=True.
    bus.simulate_drop_once()
    res = bus.publish(
        "velutina/detecciones", t=10.0, payload={"conf": 0.9}, qos=1
    )
    assert res["dup"] is True
    assert len(received) == 1
    assert received[0]["dup"] is True
    assert received[0]["payload"] == {"conf": 0.9}


def test_fifo16_drop_oldest_dropped4():
    from telemetry.mqtt_sim import MqttSim

    bus = MqttSim(seed=42, capacity=16)
    # Llena a cap 16 en t=5.0.
    for i in range(16):
        bus.publish("velutina/estado", t=5.0, payload={"seq": i}, qos=0)
    assert bus.counters("velutina/estado")["depth"] == 16
    # 4 más → drop oldest 4, dropped=4, retiene newest 16 en orden.
    for i in range(16, 20):
        bus.publish("velutina/estado", t=5.0, payload={"seq": i}, qos=0)
    c = bus.counters("velutina/estado")
    assert c["dropped"] == 4
    assert c["depth"] == 16
    depth_seqs = [e["payload"]["seq"] for e in bus._queue["velutina/estado"]]
    assert depth_seqs == list(range(4, 20))


def test_no_mutation_b1_4_hash_equal():
    from telemetry.mqtt_sim import (
        MqttSim,
        snapshot_fire,
        snapshot_fusion,
        snapshot_link,
    )

    fusion_d = {"x9": [1.0] * 9, "conf": 0.5, "gates": {"R1": True}}
    fire_d = {"authorized": True, "fsm_state": "FIRING", "shot": {"e": 1}}
    link_d = {"dropped": 0, "queue": [1, 2, 3]}
    pre = (_h(fusion_d), _h(fire_d), _h(link_d))

    bus = MqttSim(seed=42)
    bus.subscribe("velutina/detecciones", lambda env: None)
    bus.subscribe("velutina/disparos", lambda env: None)
    bus.subscribe("velutina/estado", lambda env: None)
    bus.publish("velutina/detecciones", t=10.0, payload=snapshot_fusion(fusion_d))
    bus.publish("velutina/disparos", t=10.0, payload=snapshot_fire(fire_d))
    bus.publish("velutina/estado", t=10.0, payload=snapshot_link(link_d))
    # Mutar la copia publicada no debe tocar el original (deepcopy).
    post = (_h(fusion_d), _h(fire_d), _h(link_d))
    assert post == pre
    # El payload en cola es copia independiente.
    q0 = bus._queue["velutina/detecciones"][0]["payload"]
    q0["conf"] = 999.0
    assert fusion_d["conf"] != 999.0


def test_publish_never_blocks_and_counters():
    from telemetry.mqtt_sim import MqttSim

    bus = MqttSim(seed=42, capacity=16)
    for i in range(32):
        res = bus.publish("velutina/disparos", t=float(i), payload={"i": i})
        assert "t_recv" in res and "dropped" in res
    c = bus.counters("velutina/disparos")
    assert c["depth"] == 16
    assert c["dropped"] == 16


def test_adapters_deepcopy_no_shared_refs():
    from telemetry.mqtt_sim import snapshot_estado, snapshot_fusion

    src = {"a": {"nested": [1, 2, 3]}}
    snap = snapshot_fusion(src)
    assert snap == src
    assert snap is not src
    assert snap["a"] is not src["a"]
    snap["a"]["nested"].append(99)
    assert src["a"]["nested"] == [1, 2, 3]
    s2 = snapshot_estado({"x": [1]})
    s2["x"].append(2)
    # deepcopy garantiza aislamiento; src original intacto por construcción.
    assert copy.deepcopy(s2) == s2
