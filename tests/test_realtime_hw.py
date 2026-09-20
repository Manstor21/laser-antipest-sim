"""RED 1.4 — HwInterlock veto puro <100us (must fail before impl, PR1)."""
import pathlib
from firmware.interlock_hw.hw_interlock import HwInputs, HwInterlock


def test_veto_hw_r6_active_blocks_even_if_sw_authorizes():
    inp = HwInputs(enable_sw=True, r6_hw=True, shutter_fb_closed=True, wdog_hw_ok=True)
    assert HwInterlock.eval(inp, t=1.0) is False


def test_all_clear_allows_fire():
    inp = HwInputs(enable_sw=True, r6_hw=False, shutter_fb_closed=True, wdog_hw_ok=True)
    assert HwInterlock.eval(inp, t=1.0) is True


def test_each_veto_condition_triangulation():
    base = dict(enable_sw=True, r6_hw=False, shutter_fb_closed=True, wdog_hw_ok=True)
    assert HwInterlock.eval(HwInputs(**{**base, "enable_sw": False}), t=2.0) is False
    assert HwInterlock.eval(HwInputs(**{**base, "shutter_fb_closed": False}), t=2.0) is False
    assert HwInterlock.eval(HwInputs(**{**base, "wdog_hw_ok": False}), t=2.0) is False
    # Combinational: same inputs, different t → same output (pure, no hidden state).
    inp = HwInputs(**base)
    assert HwInterlock.eval(inp, t=3.0) is True
    assert HwInterlock.eval(inp, t=3.00001) is True  # 10us poll stability


def test_hw_module_stays_import_free_from_sw():
    src = pathlib.Path(__file__).resolve().parents[1].joinpath(
        "firmware", "interlock_hw", "hw_interlock.py").read_text(encoding="utf-8")
    for banned in ("fire_controller", "fire_pipeline", "fusion_pipeline",
                   "ekf_tracker", "galvo_sim", "shutter_sim", "time.sleep", "time.time"):
        assert banned not in src
