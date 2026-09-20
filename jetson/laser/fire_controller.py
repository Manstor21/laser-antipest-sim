"""FSM + duty/energy/burst budget for 450nm sim (SIMULATED, 0€, no HW).

States: IDLE -> ARMED (authorize) -> FIRING (settle ok) -> SAFE (any fault).
Any fault forces SAFE with sim-time latency <10ms; emission stops.
Flags: laser.enabled=false default; open-shutter = SIMULATED Class 4.
"""

from collections import deque

SIMULATED = True
LASER_ENABLED_DEFAULT = False

DUTY_LIMIT = 0.001  # 0.1% sliding 1s window
WINDOW_S = 1.0
BURST_MAX = 5
COOLDOWN_S = 2.0
PEAK_W = 1000.0  # 1kW peak sim -> 3us = 3mJ
WIDTH_MIN_US = 1.0
WIDTH_MAX_US = 5.0


def pulse_energy_j(peak_w, width_us):
    """Sim pulse energy E = Ppeak * tau. 1kW x 3us = 3mJ."""
    return float(peak_w) * float(width_us) * 1e-6


class DutyMeter:
    """Sliding 1s duty meter; blocks when duty >0.1%, latched until decay."""

    def __init__(self, limit=DUTY_LIMIT, window_s=WINDOW_S):
        self.limit = float(limit)
        self.window_s = float(window_s)
        self._events = deque()  # (t, width_s)

    def _prune(self, t):
        cutoff = float(t) - self.window_s
        ev = self._events
        while ev and ev[0][0] <= cutoff:
            ev.popleft()

    def duty(self, t):
        self._prune(float(t))
        return sum(w for _, w in self._events) / self.window_s

    def request(self, t, width_us):
        t = float(t)
        width_s = float(width_us) * 1e-6
        if width_s <= 0:
            return False
        self._prune(t)
        on_time = sum(w for _, w in self._events)
        if (on_time + width_s) / self.window_s > self.limit:
            return False
        self._events.append((t, width_s))
        return True


class FireController:
    """IDLE->ARMED->FIRING->SAFE FSM with burst/cooldown + duty + energy."""

    def __init__(self, duty_limit=DUTY_LIMIT, window_s=WINDOW_S,
                 burst_max=BURST_MAX, cooldown_s=COOLDOWN_S, peak_w=PEAK_W):
        self.state = "IDLE"
        self.emission_on = False
        self.safe_enter_t = None
        self.last_fault = None
        self.duty = DutyMeter(limit=duty_limit, window_s=window_s)
        self.burst_max = int(burst_max)
        self.cooldown_s = float(cooldown_s)
        self.peak_w = float(peak_w)
        self.burst_count = 0
        self.burst_start_t = None
        self.burst_total_j = 0.0
        self.pulse_count = 0

    def try_arm(self, authorized, t):
        if self.state == "IDLE" and bool(authorized):
            self.state = "ARMED"
        return self.state

    def start_firing(self, settle_ok, t):
        if self.state == "ARMED" and bool(settle_ok):
            self.state = "FIRING"
            self.emission_on = True
        return self.state

    def fault(self, t, reason="fault"):
        self.state = "SAFE"
        self.emission_on = False
        self.safe_enter_t = float(t)
        self.last_fault = str(reason)
        return self.state

    def reset(self, t=None):
        self.state = "IDLE"
        self.emission_on = False
        self.burst_count = 0
        self.burst_start_t = None
        self.last_fault = None
        return self.state

    def _burst_allows(self, t):
        if self.burst_count < self.burst_max:
            return True
        if self.burst_start_t is None:
            return False
        if float(t) >= self.burst_start_t + self.cooldown_s:
            self.burst_count = 0
            self.burst_start_t = None
            return True
        return False

    def request_pulse(self, t, width_us):
        t = float(t)
        width = float(width_us)
        if self.state != "FIRING" or not self.emission_on:
            return False, 0.0
        if not (WIDTH_MIN_US <= width <= WIDTH_MAX_US):
            return False, 0.0
        if not self._burst_allows(t):
            return False, 0.0
        if not self.duty.request(t=t, width_us=width):
            return False, 0.0
        energy_j = pulse_energy_j(self.peak_w, width)
        if self.burst_start_t is None:
            self.burst_start_t = t
        self.burst_count += 1
        self.burst_total_j += energy_j
        self.pulse_count += 1
        return True, energy_j
