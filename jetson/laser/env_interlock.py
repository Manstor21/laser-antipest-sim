"""T/H + material fire-ignition interlock (SIMULATED, sim-only 0EUR, no HW).

Fail-safe: missing/stale T/H or unknown material inhibits (doubt->no-fire).
Thresholds sim-only: T>35C or RH<20% or (T>30C and RH<30%) inhibits.
"""

SIMULATED = True

TEMP_MAX_C = 35.0
RH_MIN_PCT = 20.0
TEMP_WARN_C = 30.0
RH_WARN_PCT = 30.0

MATERIAL_MAP = {
    "dry_grass": {"ignition_risk": "high", "allows_fire": True, "cooldown_mult": 2.0},
    "dry_straw": {"ignition_risk": "high", "allows_fire": True, "cooldown_mult": 2.0},
    "dry_wood": {"ignition_risk": "medium", "allows_fire": True, "cooldown_mult": 1.5},
    "green_leaf": {"ignition_risk": "low", "allows_fire": True, "cooldown_mult": 1.0},
    "soil": {"ignition_risk": "low", "allows_fire": True, "cooldown_mult": 1.0},
    "concrete": {"ignition_risk": "low", "allows_fire": True, "cooldown_mult": 1.0},
    "metal": {"ignition_risk": "low", "allows_fire": True, "cooldown_mult": 1.0},
    "wet": {"ignition_risk": "low", "allows_fire": True, "cooldown_mult": 1.0},
}


def th_inhibit(temp_c, rh_pct):
    """True when T/H conditions inhibit fire (fail-safe on missing input)."""
    if temp_c is None or rh_pct is None:
        return True
    try:
        temp = float(temp_c)
        rh = float(rh_pct)
    except (TypeError, ValueError):
        return True
    if temp > TEMP_MAX_C:
        return True
    if rh < RH_MIN_PCT:
        return True
    if temp > TEMP_WARN_C and rh < RH_WARN_PCT:
        return True
    return False


def material_allows(material):
    """True only for known materials whose map entry allows fire."""
    if not isinstance(material, str):
        return False
    entry = MATERIAL_MAP.get(material)
    if entry is None:
        return False
    return bool(entry.get("allows_fire", False))


def env_clear(temp_c, rh_pct, material):
    """True when T/H clear AND material allows fire (SIMULATED)."""
    if th_inhibit(temp_c, rh_pct):
        return False
    return material_allows(material)
