"""RED test 4.1 — goldens bee-heavy precision>=0.95, cero abejas (must fail before impl)."""
from pathlib import Path

import numpy as np

from jetson.fusion.fusion_pipeline import FusionPipeline
from sim.ros2.bag_replay import replay_scenario

ROOT = Path(__file__).resolve().parents[1]
GOLDENS = ROOT / "tests" / "goldens"

K = [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]
R = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
T = [0.0, 0.0, 0.0]


def _pipeline():
    return FusionPipeline(K=K, R=R, t=T, dt=0.1)


def test_golden_manifests_exist():
    for name in ("bee_heavy.yaml", "rain.yaml", "glare.yaml", "occlusion.yaml", "dusk.yaml"):
        assert (GOLDENS / name).exists(), f"missing golden {name}"


def test_bee_heavy_precision_and_zero_bee_promotion():
    # 80% bees + dusk/glare variants; doubt->bee so bee FP must be 0.
    cases = replay_scenario("bee_heavy", seed=7)
    assert len(cases) >= 50
    bees = [c for c in cases if c["truth"] == "bee"]
    assert len(bees) / len(cases) >= 0.79  # 80% bee-heavy
    tp = fp = 0
    bees_promoted = 0
    for c in cases:
        pipe = _pipeline()
        pipe.initiate(c["z0"], c["z1"])
        fused = None
        for frame in c["frames"]:
            fused = pipe.step(**frame)
        assert "gates" in fused and "conf" in fused
        if fused["promoted"]:
            if c["truth"] == "velutina":
                tp += 1
            else:
                fp += 1
                bees_promoted += 1
    denom = tp + fp
    precision = tp / denom if denom else 0.0
    assert bees_promoted == 0, f"{bees_promoted} bees promoted (must be 0)"
    assert precision >= 0.95, f"precision {precision:.3f} < 0.95"


def test_adverse_variants_keep_zero_bee_promotion():
    for variant in ("rain", "glare", "occlusion", "dusk"):
        cases = replay_scenario(variant, seed=11)
        assert len(cases) >= 10, f"{variant} golden too small"
        for c in cases:
            pipe = _pipeline()
            pipe.initiate(c["z0"], c["z1"])
            fused = None
            for frame in c["frames"]:
                fused = pipe.step(**frame)
            if c["truth"] == "bee":
                assert fused["promoted"] is False, f"{variant}: bee promoted"
            else:
                assert fused["promoted"] is True, f"{variant}: velutina missed"
