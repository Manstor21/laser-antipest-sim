"""RED 1.1 — VirtualMcu tick 1ms orden + WCET (must fail before impl, PR1)."""
from firmware.mcu.virtual_mcu import ORDER, TICK_S, Task, VirtualMcu


def test_order_and_tick():
    calls = []
    mcu = VirtualMcu(tasks=[
        Task(name="control", prio=3, wcet_s=0.0002, fn=lambda t: calls.append("control")),
        Task(name="safety", prio=1, wcet_s=0.0001, fn=lambda t: calls.append("safety")),
        Task(name="monitor", prio=5, wcet_s=0.0001, fn=lambda t: calls.append("monitor")),
        Task(name="link_rx", prio=2, wcet_s=0.0001, fn=lambda t: calls.append("link_rx")),
        Task(name="link_tx", prio=4, wcet_s=0.0001, fn=lambda t: calls.append("link_tx")),
    ])
    res = mcu.step(t=0.0)
    assert calls[0] == "safety"
    assert calls[-1] == "monitor"
    assert calls == ["safety", "link_rx", "control", "link_tx", "monitor"]
    assert res["t_next"] == 0.0 + TICK_S
    assert TICK_S == 0.001
    assert ORDER == ["safety", "link_rx", "control", "link_tx", "monitor"]


def test_wcet_overrun_counted_no_block():
    calls = []

    def slow(t):
        # Simulate work exceeding WCET without sleeping (spin on sim-time accounting).
        calls.append("control")
        return 0.0009  # consumed_s > wcet_s=0.0002 signals overrun

    mcu = VirtualMcu(tasks=[
        Task(name="safety", prio=1, wcet_s=0.0001, fn=lambda t: calls.append("safety")),
        Task(name="link_rx", prio=2, wcet_s=0.0001, fn=lambda t: calls.append("link_rx")),
        Task(name="control", prio=3, wcet_s=0.0002, fn=slow),
        Task(name="link_tx", prio=4, wcet_s=0.0001, fn=lambda t: calls.append("link_tx")),
        Task(name="monitor", prio=5, wcet_s=0.0001, fn=lambda t: calls.append("monitor")),
    ])
    res = mcu.step(t=0.001)
    assert res["overruns"] == 1
    # Overrun must not block remaining tasks.
    assert calls == ["safety", "link_rx", "control", "link_tx", "monitor"]
    assert res["executed"] == ["safety", "link_rx", "control", "link_tx", "monitor"]


def test_tick_advances_one_ms_per_step():
    mcu = VirtualMcu(tasks=[
        Task(name="safety", prio=1, wcet_s=0.0001, fn=lambda t: None),
    ])
    r1 = mcu.step(t=0.005)
    r2 = mcu.step(t=r1["t_next"])
    assert r1["t_next"] == 0.006
    assert r2["t_next"] == 0.007
    assert r2["overruns"] == 0
