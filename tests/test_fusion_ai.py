"""RED 3.1b — PR3 fusion wiring R3/R4 desde mask + voto AND + fallback (falla sin wiring)."""
import numpy as np


def _velutina_mask():
    # elipse alargada brillante: thorax oscuro + banda -> R3/R4 pass
    mask = np.zeros((320, 320), dtype=np.uint8)
    yy, xx = np.mgrid[0:320, 0:320]
    ellipse = ((xx - 160) / 60.0) ** 2 + ((yy - 160) / 25.0) ** 2 <= 1.0
    mask[ellipse] = 255
    return mask


def test_mask_measures_r3_r4_for_fusion():
    from jetson.fusion import fusion_pipeline as fp

    assert hasattr(fp, "measure_r3_r4_from_mask"), "step() debe medir R3/R4 desde mask"
    feats = fp.measure_r3_r4_from_mask(_velutina_mask())
    for k in ("thorax_v", "band_h", "band_s", "wingspan", "body"):
        assert k in feats, f"missing {k}"
    assert 1.6 <= feats["wingspan"] / feats["body"] <= 2.4
    assert feats["thorax_v"] < 60 and 15 <= feats["band_h"] <= 45 and feats["band_s"] > 80


def test_step_ai_and_gate_with_vote_and_fallback():
    from jetson.fusion.fusion_pipeline import FusionPipeline

    K = [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]
    R = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    T = [0.0, 0.0, 0.0]
    pipe = FusionPipeline(K=K, R=R, t=T, dt=0.1)
    assert hasattr(pipe, "step_with_ai"), "FusionPipeline.step_with_ai(mask, Pv, Pb, track_id) requerido"
    pipe.initiate([0.0, 0.0, 3.0], [0.5, 0.0, 3.0])
    mask = _velutina_mask()
    out = None
    # 3 frames mismo track: 2x Pv>=0.995 -> voto promueve + R3/R4 desde mask
    for pv in (0.997, 0.998, 0.500):
        out = pipe.step_with_ai(
            lidar_xyz=np.array([1.0, 0.0, 3.0]),
            v_radial=1.5,
            mask=mask,
            p_velutina=pv,
            p_bee=0.0004,
            track_id="t-ai",
            longest_axis_mm=30.0,
            sideband_hz=130.0,
        )
    assert out["ai_vote"] is True
    assert out["gates"]["R3"] is True and out["gates"]["R4"] is True
    assert out["promoted"] is True
    # veto bee: un frame Pb alto bloquea aunque Pv alto
    pipe2 = FusionPipeline(K=K, R=R, t=T, dt=0.1)
    pipe2.initiate([0.0, 0.0, 3.0], [0.5, 0.0, 3.0])
    out2 = None
    for pv, pb in ((0.997, 0.0004), (0.998, 0.0004), (0.990, 0.050)):
        out2 = pipe2.step_with_ai(
            lidar_xyz=np.array([1.0, 0.0, 3.0]),
            v_radial=1.5,
            mask=mask,
            p_velutina=pv,
            p_bee=pb,
            track_id="t-veto",
            longest_axis_mm=30.0,
            sideband_hz=130.0,
        )
    assert out2["ai_vote"] is False
    assert out2["promoted"] is False
    # fallback: >5 misses -> full-frame una vez + log
    pipe3 = FusionPipeline(K=K, R=R, t=T, dt=0.1)
    pipe3.initiate([0.0, 0.0, 3.0], [0.5, 0.0, 3.0])
    last = None
    for _ in range(7):
        last = pipe3.step_with_ai(
            lidar_xyz=None,
            v_radial=None,
            mask=None,
            p_velutina=0.10,
            p_bee=0.40,
            track_id="t-miss",
            longest_axis_mm=30.0,
            sideband_hz=0.0,
        )
    assert last["roi"]["fallback"] is True
    assert last["roi"]["w"] == 640
    assert "fallback" in str(last.get("reason", "")).lower() or last.get("fallback_logged") is True
