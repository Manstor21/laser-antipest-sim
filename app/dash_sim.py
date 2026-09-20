"""Dashboard sim pull sobre bus (SIMULATED, sim-only 0EUR).

Design #171: `app/dash_sim.py` pull en `refresh(t)`: lee últimas copias
por tópico desde `telemetry/mqtt_sim.MqttSim`, render JSON+ASCII,
`NO-DATA` si tópico vacío. Sim-time `t` explícito float, seed42
determinista vía bus. Nunca muta B1-4 (solo lee copias del bus).

Contrato:
  DashSim(bus).refresh(t) -> {"json": dict, "ascii": str}
"""

import copy
import json

SIMULATED = True

NO_DATA = "NO-DATA"

_TOPIC_TO_SECTION = {
    "velutina/detecciones": "detecciones",
    "velutina/disparos": "disparos",
    "velutina/estado": "estado",
}

_SECTIONS = ("detecciones", "disparos", "estado")


class DashSim:
    """Dashboard pull determinista sobre copias del bus."""

    def __init__(self, bus):
        self.bus = bus

    def _latest(self, topic):
        q = self.bus._queue.get(topic, [])
        if not q:
            return None
        return copy.deepcopy(q[-1]["payload"])

    def refresh(self, t):
        """Lee últimas copias por tópico y renderiza JSON+ASCII."""
        t = float(t)
        data = {}
        for topic, section in _TOPIC_TO_SECTION.items():
            payload = self._latest(topic)
            data[section] = payload if payload is not None else NO_DATA
        lines = [f"[t={t:.2f}] VELUTINA DASH (SIMULATED)"]
        for section in _SECTIONS:
            payload = data[section]
            if payload == NO_DATA:
                lines.append(f"{section} {NO_DATA}")
            else:
                blob = json.dumps(payload, sort_keys=True, default=str)
                lines.append(f"{section} {blob}")
        return {"json": data, "ascii": "\n".join(lines)}
