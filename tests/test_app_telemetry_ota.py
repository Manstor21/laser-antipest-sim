"""PR3 RED — OTA sim versionado (SIMULATED, sim-only 0EUR).

Spec: sdd/velutina-app-telemetry/spec #170 (ota-sim-update).
Design: #171 — telemetry/ota_sim.py store {current,N-1,rejected};
  promote() exige export_onnx.check()+goldens rain/glare/occlusion
  (opset 12, IoU>=0.80); fail->rollback N-1.
Tasks: #173 Phase 2 (2.6 RED ota parte, 2.7 impl).

Sim-time t explícito float, SIMULATED=True, 0EUR sin binarios.
Goldens reutilizan jetson.ai.detector.predict + metrics.iou.
"""

import numpy as np


def _crops():
    base = np.zeros((320, 320, 3), dtype=np.uint8)
    base[140:180, 120:200, 0] = 200
    rng = np.random.default_rng(11)
    rain = np.clip(base.astype(int) + (
        rng.random((320, 320, 3)) * 25).astype(int), 0, 255).astype(np.uint8)
    veil = np.zeros((320, 320, 3), dtype=np.uint8)
    veil[0:60, 0:320, :] = 40
    glare = np.clip(base.astype(int) + veil.astype(int), 0, 255).astype(np.uint8)
    occ = base.copy()
    occ[150:180, 150:200, :] = 0
    return base, {"rain": rain, "glare": glare, "occlusion": occ}


def _live_goldens():
    from jetson.ai import detector, metrics

    base, crops = _crops()
    ref = detector.predict(base)["mask"] > 0
    out = {}
    for name, crop in crops.items():
        out[name] = float(metrics.iou(detector.predict(crop)["mask"] > 0, ref))
    return out


def test_ota_simulated_flag():
    from telemetry import ota_sim

    assert ota_sim.SIMULATED is True


def test_promote_green_v2_becomes_current():
    from telemetry import ota_sim

    goldens = _live_goldens()
    assert all(v >= 0.80 for v in goldens.values())
    ota = ota_sim.OtaSim(current="v1")
    ok = ota.promote("v2", {"opset": 12, "goldens": goldens}, t=1.0)
    assert ok is True
    assert ota.current == "v2"
    assert ota.previous == "v1"  # N-1 retenido


def test_rollback_on_occlusion_fail_stays_v2():
    from telemetry import ota_sim

    goldens = _live_goldens()
    ota = ota_sim.OtaSim(current="v1")
    assert ota.promote("v2", {"opset": 12, "goldens": goldens}, t=1.0) is True
    bad = dict(goldens)
    bad["occlusion"] = 0.50  # golden roto
    ok = ota.promote("v3", {"opset": 12, "goldens": bad}, t=2.0)
    assert ok is False
    assert ota.current == "v2"
    assert "v3" in ota.rejected
    assert ota.rollback(t=2.0) == "v2"
    reasons = " ".join(e.get("reason", "") for e in ota.log)
    assert "occlusion" in reasons


def test_opset_mismatch_rejected():
    from telemetry import ota_sim

    goldens = _live_goldens()
    ota = ota_sim.OtaSim(current="v1")
    ok = ota.promote("v9", {"opset": 11, "goldens": goldens}, t=1.0)
    assert ok is False
    assert ota.current == "v1"
    assert "v9" in ota.rejected


def test_onnx_path_gate_uses_export_check(tmp_path):
    from telemetry import ota_sim

    goldens = _live_goldens()
    ckpt = tmp_path / "m.ckpt"
    ckpt.write_bytes(b"fake-ckpt")
    good = tmp_path / "good.onnx"
    from jetson.ai.export import export_onnx
    export_onnx.export_ckpt(str(ckpt), str(good))
    ota = ota_sim.OtaSim(current="v1")
    ok = ota.promote("v2", {"opset": 12, "goldens": goldens,
                            "onnx_path": str(good)}, t=1.0)
    assert ok is True
    assert ota.current == "v2"
    # Artefacto con opset manipulado -> check falla -> reject.
    import yaml
    meta = export_onnx.read_meta(str(good))
    meta["opset"] = 11
    bad = tmp_path / "bad.onnx"
    with open(bad, "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f)
    ota2 = ota_sim.OtaSim(current="v1")
    ok2 = ota2.promote("v2", {"opset": 12, "goldens": goldens,
                              "onnx_path": str(bad)}, t=1.0)
    assert ok2 is False
    assert ota2.current == "v1"
