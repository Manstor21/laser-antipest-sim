"""RED 2.3 — galvo settle/abort timestamps sin cambiar dinamica (PR2)."""


def test_galvo_settle_timestamp_exposed():
    from firmware.galvo_shutter.galvo_sim import GalvoSim

    g = GalvoSim()
    g.set_target(0.05, -0.03)
    for _ in range(2000):
        g.step()
        if g.settled():
            break
    assert g.settled() is True
    ts = g.settle_timestamp()
    assert ts is not None
    assert ts >= 0.0


def test_galvo_abort_timestamp_exposed():
    from firmware.galvo_shutter.galvo_sim import GalvoSim

    g = GalvoSim()
    g.set_target(15.0, 0.0)
    g.step()
    assert g.abort_required() is True
    ts = g.abort_timestamp()
    assert ts is not None
    assert ts >= 0.0


def test_galvo_dynamics_unchanged_triangulate():
    from firmware.galvo_shutter.galvo_sim import GalvoSim

    g = GalvoSim()
    g.set_target(0.0, 0.0)
    for _ in range(10):
        g.step()
    assert g.tracking_error_mm() == 0.0
    assert g.settle_timestamp() is not None
