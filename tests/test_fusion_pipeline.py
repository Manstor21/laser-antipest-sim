"""RED test 3.1 — wiring ekf_tracker->gate_chain->/fusion/track (must fail before impl)."""
import numpy as np

from jetson.fusion.fusion_pipeline import FusionPipeline

K = [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]
R = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
T = [0.0, 0.0, 0.0]

VELUTINA_HSV = {"thorax_v": 40.0, "band_h": 30.0, "band_s": 120.0}
BEE_HSV = {"thorax_v": 150.0, "band_h": 30.0, "band_s": 40.0}


def _pipeline():
    return FusionPipeline(K=K, R=R, t=T, dt=0.1)


def test_pipeline_emits_fused_track_velutina():
    pipe = _pipeline()
    pipe.initiate([0.0, 0.0, 3.0], [0.5, 0.0, 3.0])
    fused = None
    for i in range(5):
        x = 0.5 + 5.0 * 0.1 * (i + 1)
        # keep range <= 5 m so it stays in lidar band
        xyz = [min(x, 3.5), 0.0, 3.0]
        true = np.asarray(xyz)
        vr = float(5.0 * xyz[0] / float(np.linalg.norm(true)))
        fused = pipe.step(
            lidar_xyz=true + np.random.default_rng(7 + i).normal(0, 0.010, 3),
            v_radial=vr,
            hsv=VELUTINA_HSV,
            longest_axis_mm=30.0,
            wingspan=40.0,
            body=20.0,
            sideband_hz=130.0,
        )
    assert fused is not None
    assert "x9" in fused and "conf" in fused
    assert "gates" in fused and set(fused["gates"]) >= {"R1", "R2", "R3", "R4"}
    assert "roi" in fused and fused["roi"]["w"] in (320, 640)
    assert fused["promoted"] is True
    assert fused["conf"] > 0.5


def test_pipeline_bee_rejected_bee_safe():
    pipe = _pipeline()
    pipe.initiate([0.0, 0.0, 3.0], [0.5, 0.0, 3.0])
    fused = None
    for i in range(5):
        xyz = [min(0.5 + 5.0 * 0.1 * (i + 1), 3.5), 0.0, 3.0]
        true = np.asarray(xyz)
        vr = float(5.0 * xyz[0] / float(np.linalg.norm(true)))
        fused = pipe.step(
            lidar_xyz=true,
            v_radial=vr,
            hsv=BEE_HSV,  # amber-uniform -> R3 fail
            longest_axis_mm=30.0,
            wingspan=40.0,
            body=20.0,
            sideband_hz=130.0,
        )
    assert fused["promoted"] is False
    assert fused["gates"]["R3"] is False
    assert "bee-safe" in fused["reason"]


def test_pipeline_coasts_on_lidar_dropout():
    pipe = _pipeline()
    pipe.initiate([0.0, 0.0, 3.0], [0.5, 0.0, 3.0])
    fused = pipe.step(
        lidar_xyz=None,  # occlusion: coast on predict only
        v_radial=None,
        hsv=VELUTINA_HSV,
        longest_axis_mm=30.0,
        wingspan=40.0,
        body=20.0,
        sideband_hz=130.0,
    )
    assert "x9" in fused and "roi" in fused
    assert fused["roi"]["fallback"] is False  # single miss, no fallback yet
