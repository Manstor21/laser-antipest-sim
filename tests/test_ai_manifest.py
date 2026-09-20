"""RED 1.1 — PR1 dataset+manifest+thresholds (must fail before impl)."""
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "sim" / "dataset" / "dataset_manifest.yaml"
THRESHOLDS = ROOT / "jetson" / "ai" / "thresholds.yaml"


def _load_manifest():
    assert MANIFEST.exists(), f"missing {MANIFEST}"
    with open(MANIFEST, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_manifest_has_sources_licenses_and_ratio():
    cfg = _load_manifest()
    sources = cfg.get("sources", [])
    assert len(sources) >= 4, "expected combo: vespai + roboflow + bee-neg + gazebo-synth"
    names = {s["name"] for s in sources}
    assert "vespai-protocol" in names
    assert "roboflow-filtered" in names
    assert "bee-vs-wasp-neg" in names
    assert "gazebo-synth-320" in names
    for s in sources:
        assert s.get("license"), f"source {s.get('name')} missing license"
        assert isinstance(s.get("n"), int) and s["n"] > 0
    assert "synthetic_ratio" in cfg, "manifest must report synthetic_ratio"
    assert isinstance(cfg["synthetic_ratio"], float)
    assert 0.0 < cfg["synthetic_ratio"] < 1.0
    assert cfg.get("attribution"), "CC-BY-4.0 attribution strings required"


def test_manifest_synthetic_ratio_matches_counts():
    cfg = _load_manifest()
    by_name = {s["name"]: s["n"] for s in cfg["sources"]}
    n_total = sum(by_name.values())
    n_synth = by_name["gazebo-synth-320"]
    expected = n_synth / n_total
    assert cfg["synthetic_ratio"] == pytest.approx(expected, abs=1e-6)


def test_build_batch_has_dr_variants_and_320():
    from sim.dataset import build

    batch = build.sample_batch(n=12, seed=0)
    assert len(batch) == 12
    variants = {item["variant"] for item in batch}
    assert {"rain", "glare", "occlusion"} <= variants, f"missing DR variants: {variants}"
    for item in batch:
        assert (item["width"], item["height"]) == (320, 320)
        assert item["variant"] in {"rain", "glare", "occlusion", "clean"}


def test_thresholds_gate_doubt_to_bee_and_disabled():
    assert THRESHOLDS.exists(), f"missing {THRESHOLDS}"
    with open(THRESHOLDS, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["p_velutina_min"] == pytest.approx(0.995)
    assert cfg["p_bee_max"] == pytest.approx(0.001)
    assert cfg["temperature"] > 0.0
    assert cfg["vote"]["min_frames"] == 3
    assert cfg["vote"]["min_agree"] == 2
    assert cfg["ai"]["enabled"] is False
