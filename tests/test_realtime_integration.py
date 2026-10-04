"""RED 3.1 — Integration determinism double-run + EKF dt (PR3)."""
import numpy as np


def _promoted_out(t_meas=None, t_fire=None):
    out = {
        "promoted": True,
        "ai_vote": True,
        "gates": {"R1": True, "R2": True, "R3": True, "R4": True},
    }
    if t_meas is not None:
        out["t_meas"] = float(t_meas)
    if t_fire is not None:
        out["t_fire"] = float(t_fire)
    return out


def _run_cycle(t=10.0, fusion_out=None):
    from jetson.laser.fire_pipeline import FirePipeline

    pipe = FirePipeline(
        laser_cfg={"laser": {"enabled": True, "simulated": True}},
        explicit_config=True,
    )
    out = fusion_out if fusion_out is not None else _promoted_out()
    return pipe.cycle(
        fusion_out=out,
        target_xyz=(0.05, -0.03, 3.0),
        t=t,
    )


def test_double_run_identical_trace():
    r1 = _run_cycle(t=10.0)
    r2 = _run_cycle(t=10.0)
    assert r1["authorized"] is True
    assert r2["authorized"] is True
    assert r1["t_authorize"] == r2["t_authorize"] == 10.0
    assert r1["t_fire"] == r2["t_fire"]
    assert r1["t_fire"] > r1["t_authorize"]
    assert r1["fsm_state"] == r2["fsm_state"] == "FIRING"
    assert r1["error_mm"] == r2["error_mm"]
    assert r1["shot"] == r2["shot"]
    assert r1["settle_ts"] == r2["settle_ts"]


def test_double_run_split_stages_identical_link_delay():
    from jetson.laser.fire_pipeline import LINK_AUTH_BYTES, FirePipeline

    def _split(t=11.0):
        pipe = FirePipeline(
            laser_cfg={"laser": {"enabled": True, "simulated": True}},
            explicit_config=True,
        )
        msg = pipe.jetson_step(
            _promoted_out(), target_xyz=(0.05, -0.03, 3.0), t=t,
        )
        res = pipe.mcu_step(msg, t_fire=msg["t_recv"])
        return msg, res

    m1, r1 = _split()
    m2, r2 = _split()
    assert m1 == m2
    assert r1["t_fire"] == r2["t_fire"]
    # UART115200 32B deterministic delay: 32*8/115200 + 5us.
    expected = LINK_AUTH_BYTES * 8.0 / 115200.0 + 5e-6
    assert m1["t_recv"] - m1["t_authorize"] == float(
        np.float64(expected)
    ) or abs((m1["t_recv"] - m1["t_authorize"]) - expected) < 1e-12
    assert r1["t_fire"] > r1["t_authorize"]


def test_ekf_dt_uses_fire_minus_meas_double_run():
    from jetson.fusion.ekf_tracker import EKFTracker

    def _pred(t_meas=7.0, t_fire=None):
        ekf = EKFTracker(dt=0.1)
        ekf.x = np.array([0, 0, 3, 5, 0, 0, 0, 0, 0], dtype=float)
        # t_fire defaults to pipeline link delay from t_meas.
        if t_fire is None:
            from jetson.laser.fire_pipeline import FirePipeline

            pipe = FirePipeline(
                laser_cfg={"laser": {"enabled": True, "simulated": True}},
                explicit_config=True,
            )
            msg = pipe.jetson_step(
                _promoted_out(t_meas=t_meas), target_xyz=(0.05, -0.03, 3.0),
                t=t_meas,
            )
            t_fire = float(msg["t_recv"])
            assert msg["t_meas"] == t_meas
        return ekf.predict_at(t_fire=t_fire, t_meas=t_meas), float(t_fire)

    p1, f1 = _pred()
    p2, f2 = _pred()
    assert f1 == f2
    assert np.allclose(p1, p2)
    # dt = t_fire - t_meas > 0 compensates link latency; x = vx*dt.
    dt = f1 - 7.0
    assert dt > 0.0
    assert abs(float(p1[0]) - 5.0 * dt) < 1e-9
