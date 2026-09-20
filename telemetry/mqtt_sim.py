"""In-memory MQTT-sim bus + read-only adapters (SIMULATED, sim-only 0EUR).

Design #171: dict{tópico:FIFO16 drop-oldest+counter}, QoS0 directo /
QoS1 con reintento+dup flag, seed42 determinista, sim-time `t` explícito.
Adapters `snapshot_*()` con `copy.deepcopy`; nunca mutan B1-4
(`jetson/fusion/*, jetson/ai/*, jetson/laser/*, sim/link/*` solo lectura;
test verifica hash pre/post igual).

Contratos:
  bus.publish(topic,t,payload,qos=0) -> {"t_recv":t,"dup":bool,"dropped":int}
  bus.subscribe(topic,cb,qos=0); bus.counters(topic) -> {"depth":int,"dropped":int}
Invariantes: `t:float` explícito; `SIMULATED=True`; payloads deepcopy antes
de publicar; el bus nunca bloquea (drop-oldest).
"""

import copy
import random

SIMULATED = True

TOPICS = (
    "velutina/detecciones",
    "velutina/disparos",
    "velutina/estado",
)

_DEFAULT_CAPACITY = 16
_DEFAULT_SEED = 42


class MqttSim:
    """Broker in-memory 3 tópicos, FIFO por tópico, QoS sim, seed42."""

    def __init__(self, capacity=_DEFAULT_CAPACITY, seed=_DEFAULT_SEED,
                 topics=TOPICS):
        self.capacity = int(capacity)
        self.seed = int(seed)
        self.topics = tuple(topics)
        self._rng = random.Random(self.seed)
        self._queue = {t: [] for t in self.topics}
        self._dropped = {t: 0 for t in self.topics}
        self._subs = {t: [] for t in self.topics}
        self._fail_next = False

    # -- test hook -----------------------------------------------------
    def simulate_drop_once(self):
        """Arma una caída de transporte para el próximo publish QoS1."""
        self._fail_next = True

    # -- pub/sub -------------------------------------------------------
    def subscribe(self, topic, cb, qos=0):
        if topic not in self._queue:
            raise ValueError(f"unknown topic {topic!r}")
        if not callable(cb):
            raise ValueError("cb must be callable")
        self._subs[topic].append({"cb": cb, "qos": int(qos)})
        return len(self._subs[topic])

    def publish(self, topic, t, payload, qos=0):
        if topic not in self._queue:
            raise ValueError(f"unknown topic {topic!r}")
        t = float(t)
        qos = int(qos)
        snap = copy.deepcopy(payload)
        entry = {"t": t, "payload": snap, "qos": qos, "dup": False}
        q = self._queue[topic]
        if len(q) >= self.capacity:
            q.pop(0)
            self._dropped[topic] += 1
        q.append(entry)
        # Entrega mismo tick (síncrona, determinista).
        dup = False
        if qos == 1 and self._fail_next:
            # Primer intento cae; retry entrega con dup=True exactamente 1 vez.
            self._fail_next = False
            dup = True
            entry["dup"] = True
        for sub in list(self._subs[topic]):
            env = {
                "t": t,
                "payload": copy.deepcopy(snap),
                "qos": qos,
                "dup": bool(dup),
            }
            sub["cb"](env)
        return {"t_recv": t, "dup": bool(dup), "dropped": int(self._dropped[topic])}

    def counters(self, topic):
        if topic not in self._queue:
            raise ValueError(f"unknown topic {topic!r}")
        return {"depth": len(self._queue[topic]),
                "dropped": int(self._dropped[topic])}


# -- read-only adapters (B1-4 nunca mutados) ---------------------------
def snapshot(obj):
    """Deepcopy genérica: aísla bus/dashboard de B1-4."""
    return copy.deepcopy(obj)


def snapshot_fusion(fusion_out):
    """Copia snapshot de `jetson/fusion/*` (step/step_with_ai out)."""
    return copy.deepcopy(fusion_out)


def snapshot_ai(ai_out):
    """Copia snapshot de `jetson/ai/*` (detector/voto/thresholds)."""
    return copy.deepcopy(ai_out)


def snapshot_fire(fire_out):
    """Copia snapshot de `jetson/laser/*` (FirePipeline.cycle out)."""
    return copy.deepcopy(fire_out)


def snapshot_link(link_stats):
    """Copia snapshot de `sim/link/*` (LinkSim stats/colas)."""
    return copy.deepcopy(link_stats)


def snapshot_estado(state):
    """Copia snapshot de estado agregado para `velutina/estado`."""
    return copy.deepcopy(state)
