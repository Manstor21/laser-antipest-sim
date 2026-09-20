"""Jitter meter per-segment sim-time (SIMULATED, sim-only 0EUR, no HW).

Segments link/scheduler/galvo on sim-time t_fire-t_authorize.
Seed fixed for determinism, no wall-clock.
"""
import math

SIMULATED = True


def _pct(sorted_vals, p):
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return float(sorted_vals[0])
    k = (n - 1) * float(p)
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return float(sorted_vals[f])
    return float(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f))


class JitterMeter:
    """Record authorize/link_rx/mcu_fire/settled spans, report p50/p99/max."""

    def __init__(self, seed=42):
        self.seed = int(seed)
        self._link = []
        self._sched = []
        self._galvo = []

    def record(self, authorize_t, link_rx_t, mcu_fire_t, settled_t):
        authorize_t = float(authorize_t)
        link_rx_t = float(link_rx_t)
        mcu_fire_t = float(mcu_fire_t)
        settled_t = float(settled_t)
        link = link_rx_t - authorize_t
        sched = mcu_fire_t - link_rx_t
        galvo = settled_t - mcu_fire_t
        self._link.append(link)
        self._sched.append(sched)
        self._galvo.append(galvo)
        return {"link": link, "scheduler": sched, "galvo": galvo}

    def _stats(self, vals):
        s = sorted(vals)
        if not s:
            return {"p50": 0.0, "p99": 0.0, "max": 0.0}
        return {"p50": _pct(s, 0.5), "p99": _pct(s, 0.99), "max": float(s[-1])}

    def report(self):
        return {
            "link": self._stats(self._link),
            "scheduler": self._stats(self._sched),
            "galvo": self._stats(self._galvo),
        }
