"""RED 2.1 — FirePipeline split jetson_step/link/mcu_step + veto HW (PR2)."""
from firmware.interlock_hw.hw_interlock import HwInputs


def _promoted_out():
    return {
        "promoted": True,
        "ai_vote": True,
        "gates": {"R1": True, "R2": True, "R3": True, "R4": True},
    }


def test_split_t_fire_gt_authorize():
    from jetson.laser.fire_pipeline import FirePipeline

    pipe = FirePipeline()
    msg = pipe.jetson_step(
        {"promoted": True, "ai_vote": True,
         "gates": {"R1": True, "R2": True, "R3": True, "R4": True}},
        target_xyz=(0.05, -0.03, 3.0), t=1.0,
    )
    assert msg["authorized"] is True
    assert msg["t_authorize"] == 1.0
    res = pipe.mcu_step(msg, t_fire=msg["t_recv"])
    assert res["t_fire"] > res["t_authorize"]
    assert res["fsm_state"] == "FIRING"
    assert res["shot"] is not None


def test_split_hw_veto_cancels_fsm():
    from jetson.laser.fire_pipeline import FirePipeline

    pipe = FirePipeline()
    msg = pipe.jetson_step(_promoted_out(), target_xyz=(0.05, -0.03, 3.0), t=2.0)
    assert msg["authorized"] is True
    veto = HwInputs(enable_sw=True, r6_hw=True,
                    shutter_fb_closed=True, wdog_hw_ok=True)
    res = pipe.mcu_step(msg, t_fire=msg["t_recv"], hw_inputs=veto)
    assert res["safe"] is True
    assert res["fsm_state"] == "SAFE"
    assert res["shot"] is None
    assert res["vetoed"] is True


def test_cycle_wrapper_keeps_compat_shape():
    from jetson.laser.fire_pipeline import FirePipeline

    pipe = FirePipeline()
    res = pipe.cycle(
        fusion_out=_promoted_out(), target_xyz=(0.05, -0.03, 3.0), t=3.0,
    )
    assert res["authorized"] is True
    assert res["fsm_state"] == "FIRING"
    assert res["t_fire"] > res["t_authorize"]
    assert res["shot"] is not None
