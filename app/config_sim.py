"""Config sim copia validada (SIMULATED, sim-only 0EUR).

Design #171: `app/config_sim.py` carga copia de
`jetson/ai/thresholds.yaml` + `jetson/fusion/config.yaml`, valida rangos,
escribe solo copia sim. Nunca escribe yaml real (checksum intacto).

Contratos:
  load_copy() -> {"thresholds": dict, "fusion": dict}
  validate(cfg, p_velutina=None) -> (verdict, errors)
  save_copy(cfg, path) -> Path (rechaza paths reales)
"""

import copy
import hashlib
from pathlib import Path

import yaml

SIMULATED = True

_ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS_REAL = _ROOT / "jetson" / "ai" / "thresholds.yaml"
FUSION_REAL = _ROOT / "jetson" / "fusion" / "config.yaml"


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_copy() -> dict:
    """Carga copia fresca de thresholds + fusion (deepcopy aislada)."""
    return {
        "thresholds": copy.deepcopy(_load_yaml(THRESHOLDS_REAL)),
        "fusion": copy.deepcopy(_load_yaml(FUSION_REAL)),
    }


def validate(cfg: dict, p_velutina=None):
    """Valida copia: rangos + veto on; doubt->bee si observado < p_min."""
    errors: list = []
    thresholds = cfg.get("thresholds", {}) if isinstance(cfg, dict) else {}
    p_min = thresholds.get("p_velutina_min")
    p_bee_max = thresholds.get("p_bee_max")
    vote = thresholds.get("vote", {}) if isinstance(
        thresholds.get("vote", {}), dict) else {}
    veto = vote.get("bee_veto")

    if not isinstance(p_min, (int, float)) or not (0 < float(p_min) <= 1):
        errors.append("p_velutina_min out of range (0,1]")
    if not isinstance(p_bee_max, (int, float)) or not (
        0 <= float(p_bee_max) < 0.5
    ):
        errors.append("p_bee_max out of range [0,0.5)")
    if veto is not True:
        errors.append("veto must stay on (vote.bee_veto=True)")

    if errors:
        return ("reject", errors)
    if p_velutina is not None:
        if float(p_velutina) < float(p_min):
            return ("bee/doubt->bee", [])
        return ("ok/velutina", [])
    return ("ok", [])


def save_copy(cfg: dict, path) -> Path:
    """Escribe copia sim; rechaza paths reales thresholds/fusion."""
    dest = Path(path).resolve()
    for real in (THRESHOLDS_REAL.resolve(), FUSION_REAL.resolve()):
        if dest == real:
            raise ValueError(f"refusing to write real config {real}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        yaml.safe_dump(copy.deepcopy(cfg), f, sort_keys=True,
                       allow_unicode=True)
    return dest


def real_checksums() -> dict:
    """SHA256 de yamls reales (para probar aislamiento)."""
    out = {}
    for key, path in (("thresholds", THRESHOLDS_REAL),
                      ("fusion", FUSION_REAL)):
        out[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out
