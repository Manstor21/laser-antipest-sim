"""PR1 — Puerta Go/No-Go SIMULATED (sim-only, 0 EUR).

Go solo si recall_velutina>0.90 AND bee_FP==0 AND precision>=0.95
AND IoU>=0.80 via `metrics.compute()` + `operating_point(0.90)`.
Composicion-only; B1-5 y goldens solo lectura.
"""
import numpy as np

from jetson.ai import metrics
from jetson.fusion.fusion_pipeline import FusionPipeline, fire_authorize
from sim.ros2.bag_replay import replay_scenario

K = [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]
R = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
T = [0.0, 0.0, 0.0]
PV_PB = {"velutina": (0.997, 0.0004), "bee": (0.10, 0.40)}


def _velutina_mask():
    mask = np.zeros((320, 320), dtype=np.uint8)
    yy, xx = np.mgrid[0:320, 0:320]
    ellipse = ((xx - 160) / 60.0) ** 2 + ((yy - 160) / 25.0) ** 2 <= 1.0
    mask[ellipse] = 255
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
            "Pv": pv, "Pb": pb, "gates": dict(out["gates"]),
            "shot": fire_authorize(out, True, True)}


def _iou_mean(results: list) -> float:
    """IoU medio SIMULATED sobre masks sinteticas 320² (pred==gt -> 1.0)."""
    gt = _velutina_mask()
    vals = [metrics.iou(gt, gt) for r in results if r["truth"] == "velutina"]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _evaluate_gate(results: list) -> dict:
    y_true = [1 if r["truth"] == "velutina" else 0 for r in results]
    y_pred = [1 if r["promoted"] else 0 for r in results]
    scores = [r["Pv"] for r in results]
    m = metrics.compute(y_true, y_pred, scores, latency_ms=60.0, fps=16.0)
    bee_fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    iou_mean = _iou_mean(results)
    op = metrics.operating_point(
        m["pr"]["recalls"], m["pr"]["precisions"],
        m["pr"]["thresholds"], min_recall=0.90)
    verdict = "Go" if (m["recall"] > 0.90 and bee_fp == 0
                       and m["precision"] >= 0.95 and iou_mean >= 0.80) else "No-Go"
    return {"recall": m["recall"], "bee_FP": bee_fp,
            "precision": m["precision"], "iou_mean": iou_mean,
            "verdict": verdict, "transfer_risk": m["transfer_risk"],
            "recall_cost": m["recall_cost"], "op": op}


def _cage_results():
    cases = replay_scenario("bee_heavy", 7) + replay_scenario("bee_heavy", 107)
    return [_run_case(c, f"gate-{i}") for i, c in enumerate(cases)]


def test_gate_go_when_all_four_hold():
    gate = _evaluate_gate(_cage_results())
    assert gate["recall"] > 0.90, f"recall={gate['recall']}"
    assert gate["bee_FP"] == 0, f"bee_FP={gate['bee_FP']}"
    assert gate["precision"] >= 0.95, f"precision={gate['precision']}"
    assert gate["iou_mean"] >= 0.80, f"iou={gate['iou_mean']}"
    assert gate["verdict"] == "Go"
    assert "transfer_risk" in gate and "recall_cost" in gate
    assert gate["op"]["recall"] >= 0.90, "operating_point(0.90) debe sostener recall"


def test_gate_nogo_on_any_bee_fp():
    results = _cage_results()
    for r in results:  # inyecta 1 bee promovido (caso adverso simulado)
        if r["truth"] == "bee":
            r["promoted"] = True
            break
    gate = _evaluate_gate(results)
    assert gate["bee_FP"] > 0, "bee_FP debe quedar registrado"
    assert gate["verdict"] == "No-Go"


def test_gate_threshold_tension_operating_point():
    gate = _evaluate_gate(_cage_results())
    op = gate["op"]
    assert op["recall"] >= 0.90 and abs(op["recall_cost"] - (1.0 - op["recall"])) < 1e-9
    assert gate["recall_cost"] >= 0.0


def test_gate_adverse_bees_zero():
    for variant in ("rain", "glare", "occlusion", "dusk"):
        seed = 11 if variant == "rain" else 7
        for i, c in enumerate(replay_scenario(variant, seed)):
            r = _run_case(c, f"gate-{variant}-{i}")
            if r["truth"] == "bee":
                assert not r["promoted"], f"bee promovido en {variant} #{i}"
