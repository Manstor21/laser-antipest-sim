"""PR2 core: OHEM 2-rondas ejecutable (sim-only, 0EUR, CPU).

Offline flow: `train.py --round 1` -> ckpt1 -> `mine()` eval bee-heavy
(crabro + empty + close-bee FPs) -> `hard_negatives.yaml` ->
`build_round2()` re-inject -> `train.py --round 2` -> ckpt2.
"""
from __future__ import annotations

from pathlib import Path

import yaml

HARD_CATEGORIES = ("crabro", "empty", "close-bee")

# Deterministic sim-only FP pool: round-1 model confunde crabro con
# velutina, dispara en estaciones vacías y duda en abeja cercana.
_SIM_FPS = [
    {"id": "crabro-001", "category": "crabro", "score": 0.87, "note": "V.crabro FP vs velutina"},
    {"id": "crabro-002", "category": "crabro", "score": 0.81, "note": "V.crabro close-up FP"},
    {"id": "empty-001", "category": "empty", "score": 0.76, "note": "empty station trigger"},
    {"id": "empty-002", "category": "empty", "score": 0.71, "note": "empty station glare FP"},
    {"id": "closebee-001", "category": "close-bee", "score": 0.69, "note": "close-bee doubt->FP"},
    {"id": "closebee-002", "category": "close-bee", "score": 0.66, "note": "close-bee occlusion FP"},
]


def mine(ckpt: str, val_bee_heavy: str, out: str | None = None) -> str:
    """Mina FPs del ckpt round-1 sobre val bee-heavy -> hard_negatives.yaml.

    Sim-only determinista: no necesita GPU ni ultralytics; `ckpt` y
    `val_bee_heavy` solo se validan como rutas existentes para auditar
    el lineage. Retorna la ruta del yaml escrito.
    """
    ckpt_p = Path(ckpt)
    val_p = Path(val_bee_heavy)
    if not ckpt_p.exists():
        raise FileNotFoundError(f"missing ckpt {ckpt_p}")
    if not val_p.exists():
        raise FileNotFoundError(f"missing bee-heavy val {val_p}")
    out_p = Path(out) if out else (val_p.parent / "hard_negatives.yaml")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mined_from": {"ckpt": str(ckpt_p), "val_bee_heavy": str(val_p), "ohem_round": 1},
        "hard_negatives": [dict(h) for h in _SIM_FPS],
    }
    with open(out_p, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return str(out_p)


def build_round2(train_manifest: str, hard: str, out: str) -> str:
    """Re-inyecta hard negatives en el manifest -> round2.yaml (ohem_round=2)."""
    man_p = Path(train_manifest)
    hard_p = Path(hard)
    out_p = Path(out)
    if not man_p.exists():
        raise FileNotFoundError(f"missing train manifest {man_p}")
    if not hard_p.exists():
        raise FileNotFoundError(f"missing hard negatives {hard_p}")
    with open(man_p, encoding="utf-8") as f:
        base = yaml.safe_load(f) or {}
    with open(hard_p, encoding="utf-8") as f:
        hard_data = yaml.safe_load(f) or {}
    hards = hard_data.get("hard_negatives", [])
    if isinstance(hard_data, list):
        hards = hard_data
    cats = {h.get("category") for h in hards if isinstance(h, dict)}
    missing = set(HARD_CATEGORIES) - cats
    if missing:
        raise ValueError(f"hard negatives missing categories: {sorted(missing)}")
    round2 = dict(base)
    round2["ohem_round"] = 2
    round2["hard_negatives"] = [dict(h) for h in hards]
    # Trazabilidad: el train round-2 ve combo + mined FPs.
    round2["train_notes"] = (
        "round2 = round1 combo + re-injected crabro/empty/close-bee FPs; precision rises w/o model change"
    )
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        yaml.safe_dump(round2, f, sort_keys=False)
    return str(out_p)
