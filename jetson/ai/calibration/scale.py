"""PR2 core: calibración T-scaling + gate doubt->bee + punto operativo PR.

T se ajusta en set disjunto (PR3); aquí el forward + gate + operating
point para que el voto temporal promueva solo con Pv>=p_min AND
Pb<=p_bee_max AND R1-R4. Los umbrales se leen de thresholds.yaml.
"""
from __future__ import annotations

import math
from pathlib import Path

import yaml

from jetson.ai.temporal import get_thresholds


def load_thresholds(path: str | None = None) -> dict:
    p = Path(path) if path else (Path(__file__).resolve().parents[1] / "thresholds.yaml")
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def softmax_with_temperature(logits: list[float], temperature: float) -> list[float]:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    scaled = [float(z) / float(temperature) for z in logits]
    m = max(scaled)
    exps = [math.exp(z - m) for z in scaled]
    s = sum(exps)
    return [e / s for e in exps]


def gate(
    p_velutina: float,
    p_bee: float,
    r1_r4_ok: bool = True,
    p_min: float | None = None,
    p_bee_max: float | None = None,
) -> bool:
    """AND-gate: Pv>=p_min AND Pb<=p_bee_max AND R1-R4; si no, doubt->bee.

    Si p_min/p_bee_max no se pasan, se leen desde thresholds.yaml (si ai.enabled=true)
    o se usan los defaults (0.995 / 0.001).
    """
    thresholds = get_thresholds()
    p_min = float(p_min) if p_min is not None else thresholds.get("p_velutina_min", 0.995)
    p_bee_max = float(p_bee_max) if p_bee_max is not None else thresholds.get("p_bee_max", 0.001)
    if not r1_r4_ok:
        return False
    if float(p_velutina) < p_min:
        return False
    if float(p_bee) > p_bee_max:
        return False
    return True


def find_operating_point(
    recalls: list[float], precisions: list[float], thresholds: list[float], min_recall: float = 0.99
) -> dict:
    """Elige el punto PR con recall>=min_recall y mayor precisión.

    recall_cost = 1 - recall (coste explícito exigido por spec).
    """
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
    return best