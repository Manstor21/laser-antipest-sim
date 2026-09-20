"""PR3 — Trazabilidad MLOps SIMULATED (sim-only, 0 EUR).

`model_version` = ckpt_sha (yolov8s-seg:COCO) + opset12 + thresholds_hash
(sha256 de `jetson/ai/thresholds.yaml`) + synth_ratio 0.357 (manifest
`gazebo-synth-320` 3000/8400). `run_log.jsonl` = una linea por decision
{t,seed,truth,gates,Pv,Pb,vote,promoted,shot,version}. Composicion-only
sobre `replay_scenario` + `FusionPipeline.step_with_ai` + `export_onnx.OPSET`.
No muta codigo B1-5 ni goldens. Etiquetado SIMULATED/NO-CERT.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from jetson.ai.export import export_onnx
from jetson.fusion.fusion_pipeline import FusionPipeline, fire_authorize
from sim.ros2.bag_replay import replay_scenario

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS_PATH = ROOT / "jetson" / "ai" / "thresholds.yaml"
MANIFEST_PATH = ROOT / "sim" / "dataset" / "dataset_manifest.yaml"

K = [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]
R = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
T = [0.0, 0.0, 0.0]
PV_PB = {"velutina": (0.997, 0.0004), "bee": (0.10, 0.40)}

CKPT_DESCRIPTOR = "yolov8s-seg:COCO"  # mirrors export_onnx MODEL+INIT
SYNTH_RATIO_PINNED = 0.357  # round(3000/8400, 3); manifest guarda float exacto
SIMULATED = True
LABEL = "SIMULATED NO-CERT (sin certificar): traza sim-only, no es certificacion"


def _ckpt_sha() -> str:
    return hashlib.sha256(CKPT_DESCRIPTOR.encode("utf-8")).hexdigest()


def _thresholds_hash() -> str:
    return hashlib.sha256(THRESHOLDS_PATH.read_bytes()).hexdigest()


def _manifest_ratio() -> float:
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return float(cfg["synthetic_ratio"])


def _version() -> dict:
    return {
        "model": "yolov8s-seg",
        "init_weights": "COCO",
        "ckpt_sha": _ckpt_sha(),
        "opset": export_onnx.OPSET,
        "thresholds_hash": _thresholds_hash(),
        "synth_ratio": SYNTH_RATIO_PINNED,
        "manifest_ratio": _manifest_ratio(),
        "simulated": SIMULATED,
    }


def _velutina_mask():
    mask = np.zeros((320, 320), dtype=np.uint8)
    yy, xx = np.mgrid[0:320, 0:320]
    mask[((xx - 160) / 60.0) ** 2 + ((yy - 160) / 25.0) ** 2 <= 1.0] = 255
    return mask


def _bee_mask():
    mask = np.zeros((320, 320), dtype=np.uint8)
    mask[10:20, 10:20] = 200
    return mask


def _run_case(case: dict, track_id: str) -> dict:
    pipe = FusionPipeline(K=K, R=R, t=T, dt=0.1)
    pipe.initiate(case["z0"], case["z1"])
    pv, pb = PV_PB[case["truth"]]
    mask = _velutina_mask() if case["truth"] == "velutina" else _bee_mask()
    out = None
    for f in case["frames"]:
        lidar = None if f["lidar_xyz"] is None else np.asarray(f["lidar_xyz"])
        out = pipe.step_with_ai(
            lidar_xyz=lidar, v_radial=f["v_radial"], mask=mask,
            p_velutina=pv, p_bee=pb, track_id=track_id,
            longest_axis_mm=f["longest_axis_mm"],
            sideband_hz=f["sideband_hz"])
    return {"truth": case["truth"], "promoted": bool(out["promoted"]),
            "Pv": pv, "Pb": pb, "vote": bool(out.get("ai_vote")),
            "gates": dict(out["gates"]),
            "shot": bool(fire_authorize(out, True, True))}


def _write_run_log(path) -> Path:
    """Escribe run_log.jsonl dual-seed N=200; cada linea referencia version."""
    ver = _version()
    cases7 = replay_scenario("bee_heavy", 7)
    cases107 = replay_scenario("bee_heavy", 107)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        idx = 0
        for seed, cases in ((7, cases7), (107, cases107)):
            for i, c in enumerate(cases):
                r = _run_case(c, f"mlops-{seed}-{i}")
                line = {"t": 1000.0 + idx * 0.1, "seed": seed,
                        "truth": r["truth"], "gates": r["gates"],
                        "Pv": r["Pv"], "Pb": r["Pb"], "vote": r["vote"],
                        "promoted": r["promoted"], "shot": r["shot"],
                        "version": {"ckpt_sha": ver["ckpt_sha"],
                                    "thresholds_hash": ver["thresholds_hash"],
                                    "opset": ver["opset"]}}
                f.write(json.dumps(line, sort_keys=True) + "\n")
                idx += 1
    return path


def test_mlops_version_opset12_and_ckpt_sha():
    ver = _version()
    assert ver["opset"] == 12, f"opset={ver['opset']} != 12"
    assert export_onnx.OPSET == 12
    assert len(ver["ckpt_sha"]) == 64, "ckpt_sha debe ser sha256 hex"
    int(ver["ckpt_sha"], 16)  # hex valido
    assert ver["ckpt_sha"] == _ckpt_sha()
    assert ver["model"] == "yolov8s-seg" and ver["init_weights"] == "COCO"


def test_mlops_thresholds_hash_matches_yaml():
    h = _thresholds_hash()
    assert len(h) == 64
    assert h == hashlib.sha256(THRESHOLDS_PATH.read_bytes()).hexdigest()
    assert _version()["thresholds_hash"] == h
    cfg = yaml.safe_load(THRESHOLDS_PATH.read_text(encoding="utf-8"))
    assert cfg["p_velutina_min"] == 0.995  # ancla al thresholds real


def test_mlops_manifest_ratio_0357():
    ratio = _manifest_ratio()
    assert ratio == 0.35714285714285715, f"manifest_ratio={ratio}"
    assert round(ratio, 3) == SYNTH_RATIO_PINNED == 0.357
    assert _version()["synth_ratio"] == 0.357
    assert _version()["manifest_ratio"] == ratio


def test_mlops_run_log_jsonl_traceable(tmp_path):
    log = _write_run_log(tmp_path / "run_log.jsonl")
    assert log.exists()
    lines = log.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 200, f"lineas={len(lines)} != 200"
    ver = _version()
    for k, raw in enumerate(lines):
        e = json.loads(raw)
        for field in ("t", "seed", "truth", "gates",
                      "Pv", "Pb", "vote", "promoted", "shot", "version"):
            assert field in e, f"linea {k} sin {field}"
        assert e["seed"] in (7, 107)
        assert e["truth"] in ("velutina", "bee")
        assert isinstance(e["gates"], dict) and set(("R1", "R2", "R3", "R4")) <= set(e["gates"])
        assert e["version"]["ckpt_sha"] == ver["ckpt_sha"]
        assert e["version"]["thresholds_hash"] == ver["thresholds_hash"]
        assert e["version"]["opset"] == 12
    assert lines[0] != lines[-1]  # traza no degenerada


def test_mlops_threshold_change_detected():
    before = _thresholds_hash()
    mutated = THRESHOLDS_PATH.read_bytes() + b"\n# bump SIMULATED\n"
    after = hashlib.sha256(mutated).hexdigest()
    assert after != before, "cambio de thresholds debe cambiar el hash"
    assert len(after) == 64


def test_mlops_simulated_no_cert_label():
    assert SIMULATED is True
    assert "SIMULATED" in LABEL and "NO-CERT" in LABEL
    assert _version()["simulated"] is True
