"""RED 1.1 — AND gate parametrized deny (must fail before impl, PR1 sim-only)."""
import pytest

from jetson.fusion.fusion_pipeline import fire_authorize
from jetson.fusion.hard_rules import gate_chain, r6_interlock


def _base_out():
    return {
        "promoted": True,
        "ai_vote": True,
        "gates": {"R1": True, "R2": True, "R3": True, "R4": True},
    }


def test_all_inputs_true_authorizes():
    out = _base_out()
    assert fire_authorize(out, interlock_clear=True, shutter_ready=True) is True


@pytest.mark.parametrize(
    "mutate",
    [
        lambda o: o.update(promoted=False),
        lambda o: o.update(ai_vote=False),
        lambda o: o.update(ai_vote=None),
        lambda o: o["gates"].update(R1=False),
        lambda o: o["gates"].update(R2=False),
        lambda o: o["gates"].update(R3=False),
        lambda o: o["gates"].update(R4=False),
    ],
    ids=["promoted", "ai_vote-False", "ai_vote-None", "R1", "R2", "R3", "R4"],
)
def test_single_input_fails_denies(mutate):
    out = _base_out()
    mutate(out)
    assert fire_authorize(out, interlock_clear=True, shutter_ready=True) is False


@pytest.mark.parametrize(
    "interlock,shutter",
    [(False, True), (True, False), (False, False)],
    ids=["interlock", "shutter", "both"],
)
def test_interlock_or_shutter_false_denies(interlock, shutter):
    out = _base_out()
    assert fire_authorize(out, interlock_clear=interlock, shutter_ready=shutter) is False


def test_missing_stale_defaults_false():
    assert fire_authorize({}, interlock_clear=True, shutter_ready=True) is False
    assert fire_authorize({"promoted": True}, interlock_clear=True, shutter_ready=True) is False
    out = _base_out()
    del out["ai_vote"]
    assert fire_authorize(out, interlock_clear=True, shutter_ready=True) is False
    assert fire_authorize(None, interlock_clear=True, shutter_ready=True) is False


def test_r6_human_blocks():
    assert r6_interlock(1.5, 10.0) is False
    assert r6_interlock(1.99, 10.0) is False
    assert r6_interlock(2.0, 10.0) is True
    assert r6_interlock(None, 10.0) is True


def test_r6_track_blocks():
    assert r6_interlock(None, 40.0) is True
    assert r6_interlock(None, 40.1) is False
    assert r6_interlock(5.0, 50.0) is False


def test_r6_blocks_promoted_target():
    out = _base_out()
    # promoted True but near-field track >40mm -> R6 fails -> no authorize
    assert r6_interlock(None, 45.0) is False
    assert fire_authorize(out, interlock_clear=r6_interlock(None, 45.0), shutter_ready=True) is False


def test_gate_chain_still_r1_r4_only():
    res = gate_chain(
        {
            "longest_axis_mm": 30.0,
            "velocity_hist": [6.0, 6.1, 5.9],
            "thorax_v": 40.0,
            "band_h": 30.0,
            "band_s": 120.0,
            "wingspan": 40.0,
            "body": 20.0,
            "sideband_hz": 130.0,
        }
    )
    assert set(res["gates"]) == {"R1", "R2", "R3", "R4"}
