"""RED test 2.3 — hard_rules R1-R4 + R5 advisory (must fail before impl)."""
from jetson.fusion.hard_rules import (
    gate_chain,
    r1_size_ok,
    r2_classify,
    r3_hsv_ok,
    r4_ratio_ok,
    r5_score,
)


def test_r1_hornet_pass_bee_leaf_reject():
    assert r1_size_ok(30.0) is True
    assert r1_size_ok(13.0) is False  # bee
    assert r1_size_ok(60.0) is False  # leaf
    assert r1_size_ok(11.9) is False
    assert r1_size_ok(40.1) is False


def test_r2_hover_vs_transit():
    assert r2_classify([0.1, 0.05, 0.12]) == "hover"
    assert r2_classify([8.0, 8.1, 7.9]) == "transit"
    assert r2_classify([8.0]) == "invalid"  # needs >= 3 frames
    assert r2_classify([0.1, 25.0, 0.2]) == "invalid"  # out of band


def test_r3_r4_gates():
    assert r3_hsv_ok(thorax_v=40, band_h=30, band_s=120) is True
    assert r3_hsv_ok(thorax_v=150, band_h=30, band_s=120) is False  # amber uniform
    assert r4_ratio_ok(wingspan=40.0, body=20.0) is True  # ratio 2.0
    assert r4_ratio_ok(wingspan=20.0, body=20.0) is False  # ratio 1.0


def test_r5_advisory_never_gates():
    assert r5_score(130.0) > r5_score(60.0)  # sideband outranks
    assert r5_score(60.0) >= 0.0  # but never negative/rejecting


def test_velutina_passes_chain():
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
    assert res["promoted"] is True
    assert all(res["gates"].values())
    assert "confidence" in res


def test_bee_doubt_blocks():
    res = gate_chain(
        {
            "longest_axis_mm": 30.0,
            "velocity_hist": [6.0, 6.1, 5.9],
            "thorax_v": 150.0,  # amber-uniform ROI → R3 fail
            "band_h": 30.0,
            "band_s": 120.0,
            "wingspan": 40.0,
            "body": 20.0,
            "sideband_hz": 130.0,
        }
    )
    assert res["promoted"] is False
    assert res["gates"]["R3"] is False
