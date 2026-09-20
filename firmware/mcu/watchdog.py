"""MCU watchdog 10ms independent (SIMULATED, sim-only 0EUR, no HW)."""

SIMULATED = True
TIMEOUT_S = 0.010


class McuWatchdog:
    """Sim-time watchdog; expired iff t - last_feed > 10ms."""

    def __init__(self, timeout_s=TIMEOUT_S):
        self.timeout_s = float(timeout_s)
        self._last = None

    def feed(self, t):
        self._last = float(t)
        return self._last

    def expired(self, t):
        if self._last is None:
            return True
        return bool(float(t) - self._last > self.timeout_s)
