"""Laser feature flags (SIMULATED, sim-only 0EUR, no cert/SIL claim).

Defaults: laser.enabled=false (no-fire unless explicitly enabled).
Open-shutter emission is SIMULATED Class 4 sim-only.
"""

SIMULATED = True
SIMULATED_LABEL = "SIMULATED"
LASER_ENABLED_DEFAULT = False
CLASS_LABEL = "SIMULATED Class 4 (open-shutter, sim-only, no cert/SIL claim)"


def laser_enabled(cfg):
    """True only when cfg explicitly sets laser.enabled=true.

    Missing/stale config defaults to False (no-fire fail-safe).
    """
    try:
        if not isinstance(cfg, dict):
            return False
        laser = cfg.get("laser")
        if not isinstance(laser, dict):
            return False
        return bool(laser.get("enabled", False))
    except (AttributeError, TypeError, ValueError):
        return False
