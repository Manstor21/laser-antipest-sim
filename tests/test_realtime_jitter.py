"""RED 1.5 — JitterMeter split tramos seed 42 (must fail before impl, PR1)."""
import random

import pytest

from sim.jitter_meter import JitterMeter


def _feed(meter, n=1000, seed=42):
    rng = random.Random(seed)
    for _ in range(n):
        a = rng.uniform(0.0, 1.0)
        link_rx = a + rng.uniform(0.002, 0.003)
        mcu_fire = link_rx + rng.uniform(0.001, 0.002)
        settled = mcu_fire + rng.uniform(0.0004, 0.0006)
        meter.record(authorize_t=a, link_rx_t=link_rx,
                     mcu_fire_t=mcu_fire, settled_t=settled)


def test_split_segments_p50_p99_max_seed42():
    meter = JitterMeter(seed=42)
    _feed(meter, n=1000, seed=42)
    rep = meter.report()
    assert set(rep.keys()) == {"link", "scheduler", "galvo"}
    for seg in ("link", "scheduler", "galvo"):
        assert set(rep[seg].keys()) == {"p50", "p99", "max"}
        assert rep[seg]["p50"] <= rep[seg]["p99"] <= rep[seg]["max"]
        assert rep[seg]["max"] > 0.0
    # Link tramo dominates galvo tramo by construction of the feed ranges.
    assert rep["link"]["p50"] > rep["galvo"]["p50"]


def test_determinism_double_run_identical():
    m1, m2 = JitterMeter(seed=42), JitterMeter(seed=42)
    _feed(m1, n=1000, seed=42)
    _feed(m2, n=1000, seed=42)
    assert m1.report() == m2.report()


def test_single_record_spans_triangulation():
    meter = JitterMeter(seed=42)
    meter.record(authorize_t=1.0, link_rx_t=1.002, mcu_fire_t=1.003, settled_t=1.004)
    rep = meter.report()
    assert rep["link"]["p50"] == pytest.approx(0.002)
    assert rep["scheduler"]["p50"] == pytest.approx(0.001)
    assert rep["galvo"]["p50"] == pytest.approx(0.001)
