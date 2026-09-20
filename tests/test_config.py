"""RED test 1.1 — config thresholds (must fail before impl)."""
from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parents[1] / "jetson" / "fusion" / "config.yaml"


def _load():
    assert CONFIG.exists(), f"missing {CONFIG}"
    with open(CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_config_has_r1_thresholds():
    cfg = _load()
    r1 = cfg["gates"]["r1_size_mm"]
    assert r1["min"] == 15
    assert r1["max"] == 35
    assert r1["tolerance_mm"] == 3
    assert r1["reject_below_mm"] == 12
    assert r1["reject_above_mm"] == 40


def test_config_has_r2_r3_r4():
    cfg = _load()
    r2 = cfg["gates"]["r2_velocity_ms"]
    assert r2["hover_max"] == 0.5
    assert r2["transit_min"] == 1.0
    assert r2["transit_max"] == 11.0
    assert r2["min_frames"] == 3
    assert r2["max_error"] == 0.3
    r3 = cfg["gates"]["r3_hsv"]
    assert r3["thorax_v_max"] == 60
    assert r3["band_h_min"] == 15
    assert r3["band_h_max"] == 45
    assert r3["band_s_min"] == 80
    r4 = cfg["gates"]["r4_ratio"]
    assert r4["min"] == 1.6
    assert r4["max"] == 2.4


def test_config_has_roi_sync_and_flag():
    cfg = _load()
    assert cfg["roi"]["size_px"] == 320
    assert cfg["roi"]["miss_fallback_frames"] == 5
    assert cfg["sync"]["rms_max_ms"] == 1.0
    assert cfg["sync"]["reproj_max_px_at_2m"] == 2.0
    assert cfg["fusion"]["enabled"] is False
    assert cfg["fusion"]["predict_ms"] == 50
