"""PR3: métricas aceptación + IoU goldens (sim-only CPU, 0EUR).

`compute` reporta recall-velutina, bee-FP en val bee-heavy, mAP proxy,
latencia/FPS@320 y flag 50-70ms como riesgo de transferencia Orin.
`iou` sostiene goldens rain/glare/occlusion (delta IoU falla si regresión).
"""
from __future__ import annotations

import math

import numpy as np


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return inter / union


def _ap_from_scores(y_true: list[int], scores: list[float]) -> float:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    tp = fp = 0
    total_pos = sum(1 for y in y_true if y == 1) or 1
    precisions = []
    for i in order:
        if y_true[i] == 1:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
    # AP proxy: media de precisiones en posiciones positivas.
    pos_prec = [precisions[k] for k, i in enumerate(order) if y_true[i] == 1]
    return float(sum(pos_prec) / len(pos_prec)) if pos_prec else 0.0


def compute(
    y_true: list[int],
    y_pred: list[int],
    scores: list[float],
    latency_ms: float = 60.0,
    fps: float = 16.0,
) -> dict:
    if not (len(y_true) == len(y_pred) == len(scores)):
        raise ValueError("y_true/y_pred/scores must align")
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    n_bee = sum(1 for t in y_true if t == 0) or 1
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    bee_fp_rate = fp / n_bee
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    ap = _ap_from_scores([int(v) for v in y_true], [float(s) for s in scores])
    # PR curve ordenada por score para el operating point.
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    recalls: list[float] = []
    precisions: list[float] = []
    thresholds: list[float] = []
    ctp = cfp = 0
    total_pos = sum(1 for t in y_true if t == 1) or 1
    for i in order:
        if y_true[i] == 1:
            ctp += 1
        else:
            cfp += 1
        recalls.append(ctp / total_pos)
        precisions.append(ctp / (ctp + cfp))
        thresholds.append(float(scores[i]))
    return {
        "recall": float(recall),
        "bee_fp_rate": float(bee_fp_rate),
        "precision": float(precision),
        "map": float(ap),
        "pr": {"recalls": recalls, "precisions": precisions, "thresholds": thresholds},
        "latency_ms": float(latency_ms),
        "fps": float(fps),
        "recall_cost": float(1.0 - recall),
        "transfer_risk": bool(50.0 <= float(latency_ms) <= 70.0),
    }


def operating_point(
    recalls: list[float],
    precisions: list[float],
    thresholds: list[float],
    min_recall: float = 0.99,
) -> dict:
    best = None
    for r, p, t in zip(recalls, precisions, thresholds):
        if r < min_recall:
            continue
        cand = {
            "threshold": float(t),
            "recall": float(r),
            "precision": float(p),
            "recall_cost": float(1.0 - r),
        }
        if best is None or cand["precision"] > best["precision"]:
            best = cand
    if best is None:
        raise ValueError("no operating point meets min_recall")
    if not math.isfinite(best["recall_cost"]):
        raise ValueError("recall_cost must be finite")
    return best


def as_table(m: dict) -> str:
    lines = [
        "# AI acceptance metrics (sim-only CPU 320)",
        f"recall: {m.get('recall')}",
        f"bee-fp: {m.get('bee_fp_rate')}",
        f"bee_fp_rate: {m.get('bee_fp_rate')}",
        f"mAP: {m.get('map')}",
        f"latency_ms: {m.get('latency_ms')}",
        f"fps: {m.get('fps')}",
        f"recall_cost: {m.get('recall_cost')}",
    ]
    return "\n".join(lines)
