"""Virtual MCU cooperative tick 1ms (SIMULATED, sim-only 0EUR, no HW).

Fixed order safety>link_rx>control>link_tx>monitor, TICK 1ms, WCET
overrun counted never blocks. Explicit sim-time `t`, no sleep/time.time.
"""

SIMULATED = True
TICK_S = 0.001
ORDER = ["safety", "link_rx", "control", "link_tx", "monitor"]


class Task:
    """Cooperative task with WCET budget."""

    def __init__(self, name, prio, wcet_s, fn):
        self.name = str(name)
        self.prio = int(prio)
        self.wcet_s = float(wcet_s)
        self.fn = fn


class VirtualMcu:
    """Discrete sim-time MCU; step(t) advances t+TICK_S."""

    def __init__(self, tasks=None, watchdog=None):
        self.tasks = list(tasks) if tasks is not None else []
        self.watchdog = watchdog
        self.overruns = 0
        self.safe = False
        self.t = 0.0

    def _ordered(self):
        rank = {name: i for i, name in enumerate(ORDER)}
        return sorted(self.tasks,
                      key=lambda tk: (rank.get(tk.name, 99), tk.prio))

    def step(self, t):
        t = float(t)
        # Safety hook: independent watchdog forces SAFE, never blocks tick.
        if self.watchdog is not None:
            try:
                if bool(self.watchdog.expired(t)):
                    self.safe = True
            except (AttributeError, TypeError, ValueError):
                self.safe = True
        executed = []
        step_overruns = 0
        for task in self._ordered():
            consumed = task.fn(t)
            executed.append(task.name)
            try:
                consumed_f = float(consumed) if consumed is not None else 0.0
            except (TypeError, ValueError):
                consumed_f = 0.0
            if consumed_f > task.wcet_s:
                step_overruns += 1
        self.overruns += step_overruns
        self.t = t + TICK_S
        return {"executed": executed, "overruns": step_overruns,
                "t_next": self.t, "safe": self.safe}
