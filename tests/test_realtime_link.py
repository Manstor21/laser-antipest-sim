"""RED 1.3 — LinkSim latencia UART + FIFO drop-oldest (must fail before impl, PR1)."""
from sim.link.link_sim import LinkSim


def test_uart_32b_latency():
    link = LinkSim(baud=115200, prop_s=5e-6, capacity=16, seed=42)
    t_send = 1.0
    payload = bytes(32)
    info = link.send(t=t_send, payload=payload)
    expected = 32 * 8 / 115200 + 5e-6
    assert info["t_recv"] == t_send + expected
    assert info["t_send"] == t_send
    # Nothing due before t_recv; delivered at t_recv.
    assert link.recv(t=t_send + expected - 1e-9) == []
    got = link.recv(t=t_send + expected)
    assert len(got) == 1
    assert got[0]["t_send"] == t_send
    assert got[0]["t_recv"] == t_send + expected


def test_fifo_full_drop_oldest_never_blocks():
    link = LinkSim(baud=115200, prop_s=5e-6, capacity=2, seed=42)
    link.send(t=0.0, payload=b"A")
    link.send(t=0.001, payload=b"B")
    # Third send with full FIFO must drop oldest, count it, never block.
    info = link.send(t=0.002, payload=b"C")
    assert info["dropped"] == 1
    assert link.dropped == 1
    got = link.recv(t=10.0)
    payloads = [bytes(p["payload"]) for p in got]
    assert payloads == [b"B", b"C"]
    assert len(got) == 2


def test_spi_param_and_empty_recv_triangulation():
    link = LinkSim(baud=10_000_000, prop_s=1e-6, capacity=16, seed=42)
    info = link.send(t=0.5, payload=bytes(10))
    expected = 10 * 8 / 10_000_000 + 1e-6
    assert info["t_recv"] == 0.5 + expected
    # Empty queue returns empty list (production code ran, queue was empty).
    fresh = LinkSim(seed=42)
    assert fresh.recv(t=0.0) == []
    assert fresh.dropped == 0
