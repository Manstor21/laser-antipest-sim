"""NC dual-channel shutter sim + 10ms watchdog (SIMULATED, 0EUR, no HW).

Normally-closed: commanded state defaults to closed. Dual channels must
agree; any command/feedback mismatch or watchdog loss >10ms forces
not-ready (caller FSM must go SAFE <10ms). Closed attenuation OD>=3.5
keeps accessible energy <1.8uJ for 1-5mJ pulses.
"""

SIMULATED = True
OD_CLOSED = 3.5
WATCHDOG_TIMEOUT_S = 0.010
ACCESSIBLE_LIMIT_J = 1.8e-6


def accessible_energy_j(pulse_j, shutter_open):
    """Accessible energy: full pulse if open, else E*10^(-OD)."""
    pulse_j = float(pulse_j)
    if bool(shutter_open):
        return pulse_j
    return pulse_j * (10.0 ** (-OD_CLOSED))


class ShutterSim:
    """NC shutter with dual-channel agreement + watchdog."""

    def __init__(self, od_closed=OD_CLOSED, watchdog_s=WATCHDOG_TIMEOUT_S):
        self.od_closed = float(od_closed)
        self.watchdog_s = float(watchdog_s)
        self.commanded_open = False  # NC default: closed
        self.ch_a_open = False
        self.ch_b_open = False
        self._stuck_open = False
        self._last_feed_t = None
        self.mismatch = False

    def command(self, close=None, open=None, t=0.0):
        """Command shutter. Prefer close=True/False; open= is alias."""
        if close is not None:
            self.commanded_open = not bool(close)
        elif open is not None:
            self.commanded_open = bool(open)
        if not self._stuck_open:
            self.ch_a_open = self.commanded_open
            self.ch_b_open = self.commanded_open
        else:
            # stuck-open: feedback stays open even when commanded closed
            self.ch_a_open = True
            self.ch_b_open = True
        self._update_mismatch()
        return self.commanded_open

    def inject_stuck_open(self, stuck):
        self._stuck_open = bool(stuck)
        if self._stuck_open:
            self.ch_a_open = True
            self.ch_b_open = True
        else:
            self.ch_a_open = self.commanded_open
            self.ch_b_open = self.commanded_open
        self._update_mismatch()
        return self._stuck_open

    def _update_mismatch(self):
        feedback_open = self.ch_a_open or self.ch_b_open
        channels_agree = self.ch_a_open == self.ch_b_open
        self.mismatch = (feedback_open != self.commanded_open) or (not channels_agree)
        return self.mismatch

    def feed_watchdog(self, t):
        self._last_feed_t = float(t)
        return self._last_feed_t

    def watchdog_ok(self, t):
        if self._last_feed_t is None:
            return False
        return (float(t) - self._last_feed_t) <= self.watchdog_s

    def feedback_closed(self):
        return not (self.ch_a_open or self.ch_b_open)

    def evaluate(self, t):
        """True = shutter ready (no mismatch, watchdog ok)."""
        self._update_mismatch()
        if self.mismatch:
            return False
        if not self.watchdog_ok(float(t)):
            return False
        return True

    def accessible_j(self, pulse_j):
        feedback_open = self.ch_a_open or self.ch_b_open
        return accessible_energy_j(pulse_j, shutter_open=feedback_open)
