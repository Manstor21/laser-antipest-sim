"""PR2 core: voto temporal 3 frames por track-id (2/3 + veto bee).

Promote iff >=min_agree frames con P(velutina)>=p_min en el mismo
track-id AND ningún frame supera p_bee_max (doubt->bee, no-fire).

Thresholds se leen desde jetson/ai/thresholds.yaml si ai.enabled=true.
Los valores en código son defaults; el YAML manda cuando está habilitado.
"""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
THRESHOLDS_PATH = REPO_ROOT / "jetson" / "ai" / "thresholds.yaml"


def _load_thresholds() -> dict:
    """Carga thresholds.yaml. Retorna dict vacío si no existe o ai.enabled=false."""
    if not THRESHOLDS_PATH.exists():
        return {}
    try:
        with open(THRESHOLDS_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if data.get("ai", {}).get("enabled") is True:
            return data
    except Exception:
        pass
    return {}


_DEFAULTS = {
    "p_velutina_min": 0.995,
    "p_bee_max": 0.001,
    "temperature": 1.5,
    "vote": {
        "min_frames": 3,
        "min_agree": 2,
        "bee_veto": True,
    },
    "ai": {"enabled": False},
    "roi_size_px": 320,
}

_thresholds_cache: dict | None = None


def get_thresholds() -> dict:
    """Retorna thresholds combinados: YAML (si enabled) sobre defaults."""
    global _thresholds_cache
    if _thresholds_cache is not None:
        return _thresholds_cache
    yaml_data = _load_thresholds()
    # Merge: defaults first, then YAML values
    merged = dict(_DEFAULTS)
    merged.update(yaml_data)
    # Deep merge for nested dicts
    for key in ("vote", "ai"):
        if key in yaml_data and isinstance(yaml_data[key], dict):
            merged[key] = {**_DEFAULTS.get(key, {}), **yaml_data[key]}
    _thresholds_cache = merged
    return merged


def reset_thresholds_cache() -> None:
    """Util para tests."""
    global _thresholds_cache
    _thresholds_cache = None


class TemporalVote:
    def __init__(
        self,
        p_min: float | None = None,
        p_bee_max: float | None = None,
        min_frames: int | None = None,
        min_agree: int | None = None,
        bee_veto: bool | None = None,
    ) -> None:
        thresholds = get_thresholds()
        vote_cfg = thresholds.get("vote", {})
        self.p_min = p_min if p_min is not None else thresholds.get("p_velutina_min", _DEFAULTS["p_velutina_min"])
        self.p_bee_max = p_bee_max if p_bee_max is not None else thresholds.get("p_bee_max", _DEFAULTS["p_bee_max"])
        self.min_frames = min_frames if min_frames is not None else vote_cfg.get("min_frames", _DEFAULTS["vote"]["min_frames"])
        self.min_agree = min_agree if min_agree is not None else vote_cfg.get("min_agree", _DEFAULTS["vote"]["min_agree"])
        self.bee_veto = bee_veto if bee_veto is not None else vote_cfg.get("bee_veto", _DEFAULTS["vote"]["bee_veto"])
        self._tracks: dict[str, list[tuple[float, float]]] = {}

    def update(self, track_id: str, p_velutina: float, p_bee: float) -> bool | None:
        buf = self._tracks.setdefault(str(track_id), [])
        buf.append((float(p_velutina), float(p_bee)))
        # Ventana deslizante de min_frames.
        if len(buf) > self.min_frames:
            buf.pop(0)
        if len(buf) < self.min_frames:
            return None
        if self.bee_veto and any(pb > self.p_bee_max for _, pb in buf):
            return False
        agree = sum(1 for pv, _ in buf if pv >= self.p_min)
        return True if agree >= self.min_agree else False

    def reset(self, track_id: str) -> None:
        self._tracks.pop(str(track_id), None)