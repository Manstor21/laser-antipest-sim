"""RED 3.1 — EKF->gate->FSM->galvo->f-theta wiring + abort 15mm (PR3 sim-only)."""
import numpy as np
import pytest

from jetson.laser.fire_pipeline import FirePipeline


def _promoted_out():
    return {
        "promoted": True,
        "ai_vote": True,
        "gates": {"R1": True, "R2": True, "R3": True, "R4": True},
    }


def test_wire_happy_path_settle_then_shot_logged():
    pipe = FirePipeline(
        laser_cfg={"laser": {"enabled": True, "simulated": True}},
        explicit_config=True,
    )
    out = _promoted_out()
    # EKF x9+50ms target near field centre (z=3m sim)
    res = pipe.cycle(
        fusion_out=out,
        target_xyz=(0.05, -0.03, 3.0),
        human_min_m=5.0,
        track_mm=10.0,
        t=100.0,
        temp_c=22.0,
        rh_pct=55.0,
        material="soil",
        energy_j=3e-3,
        spot_um=300.0,
        width_us=3.0,
    )
    assert res["authorized"] is True
    assert res["fsm_state"] == "FIRING"
    assert res["aborted"] is False
    assert res["safe"] is False
    assert res["shot"] is not None
    assert res["shot"]["fluence_Jcm2"] == pytest.approx(4.24, rel=0.05)
    assert res["shot"]["simulated"] is True


def test_wire_tracking_abort_15mm_goes_safe():
    pipe = FirePipeline(
        laser_cfg={"laser": {"enabled": True, "simulated": True}},
        explicit_config=True,
    )
    out = _promoted_out()
    # prime near centre first so galvo sits near (0,0)
    pipe.cycle(
        fusion_out=out,
        target_xyz=(0.0, 0.0, 3.0),
        human_min_m=5.0,
        track_mm=10.0,
        t=200.0,
        temp_c=22.0,
        rh_pct=55.0,
        material="soil",
        energy_j=3e-3,
        spot_um=300.0,
        width_us=3.0,
    )
    # fast 5-8m/s jump -> 15mm field error must abort to SAFE
    res = pipe.cycle(
        fusion_out=out,
        target_xyz=(15.0, 0.0, 3.0),
        human_min_m=5.0,
        track_mm=10.0,
        t=200.010,
        temp_c=22.0,
        rh_pct=55.0,
        material="soil",
        energy_j=3e-3,
        spot_um=300.0,
        width_us=3.0,
        field_override_mm=(15.0, 0.0),
    )
    assert res["aborted"] is True
    assert res["safe"] is True
    assert res["fsm_state"] == "SAFE"
    assert res["error_mm"] > 10.0


def test_wire_denied_gate_never_arms():
    pipe = FirePipeline(
        laser_cfg={"laser": {"enabled": True, "simulated": True}},
        explicit_config=True,
    )
    out = _promoted_out()
    out["gates"]["R1"] = False
    res = pipe.cycle(
        fusion_out=out,
        target_xyz=(0.05, -0.03, 3.0),
        human_min_m=5.0,
        track_mm=10.0,
        t=300.0,
        temp_c=22.0,
        rh_pct=55.0,
        material="soil",
        energy_j=3e-3,
        spot_um=300.0,
        width_us=3.0,
    )
    assert res["authorized"] is False
    assert res["fsm_state"] in ("IDLE", "SAFE")
    assert res["shot"] is None
