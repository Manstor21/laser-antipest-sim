"""Link delay + bounded FIFO (SIMULATED, sim-only 0EUR, no HW).

DI defaults: UART115200 (baud=115200, prop 5us), FIFO capacity 16 with
drop-oldest + counter, seed 42. SPI10MHz available as param (prop 1us).
delay = bytes*8/baud + prop. FIFO never blocks. Explicit sim-time `t`.
Open Q: SPI vs UART default for integration seed; FIFO 16 vs 32 under
EKF 10Hz burst (current 16 + counter, tune later).
"""
import random

SIMULATED = True


class LinkSim:
    """Sim-time link: UART115200 default / SPI10MHz param, FIFO16 drop-oldest."""

    def __init__(self, baud=115200, prop_s=5e-6, capacity=16, seed=42):
        self.baud = float(baud)
        self.prop_s = float(prop_s)
        self.capacity = int(capacity)
        self.seed = int(seed)
        self._rng = random.Random(self.seed)
        self._queue = []
        self.dropped = 0

    def delay_for(self, nbytes):
        return float(nbytes) * 8.0 / self.baud + self.prop_s

    def send(self, t, payload):
        t = float(t)
        data = bytes(payload)
        t_recv = t + self.delay_for(len(data))
        entry = {"payload": data, "t_send": t, "t_recv": t_recv}
        if len(self._queue) >= self.capacity:
            self._queue.pop(0)
            self.dropped += 1
        self._queue.append(entry)
        return {"t_send": t, "t_recv": t_recv, "dropped": self.dropped}

    def recv(self, t):
        t = float(t)
        due = [p for p in self._queue if p["t_recv"] <= t]
        if due:
            due_ids = {id(p) for p in due}
            self._queue = [p for p in self._queue if id(p) not in due_ids]
        return due
