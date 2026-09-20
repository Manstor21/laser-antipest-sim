"""f-theta optics sim f=100mm, spot 200-500um (SIMULATED, 0EUR, no HW).

Fluence F = E / (pi*(d/2)^2). 1mJ@500um -> ~0.51 J/cm2,
5mJ@200um -> ~15.9 J/cm2. Field +/-35mm. Shots logged in memory.
"""

import math

SIMULATED = True
FOCAL_MM = 100.0
FIELD_HALF_MM = 35.0
SPOT_MIN_UM = 200.0
SPOT_MAX_UM = 500.0

SHOT_LOG = []


def _check_spot(spot_um):
    spot = float(spot_um)
    if not (SPOT_MIN_UM <= spot <= SPOT_MAX_UM):
        raise ValueError(f"spot_um {spot} outside [{SPOT_MIN_UM},{SPOT_MAX_UM}]")
    return spot


def fluence_Jcm2(energy_j, spot_um):
    """Per-pulse fluence in J/cm2 for a top-hat spot of diameter spot_um."""
    spot = _check_spot(spot_um)
    energy = float(energy_j)
    if energy < 0:
        raise ValueError("energy_j must be >= 0")
    d_cm = spot * 1e-4  # um -> cm
    area_cm2 = math.pi * (d_cm / 2.0) ** 2
    return energy / area_cm2


def evaluate_shot(energy_j, spot_um, x_mm=0.0, y_mm=0.0):
    """Validate field + spot, compute fluence, append to SHOT_LOG."""
    x = float(x_mm)
    y = float(y_mm)
    if abs(x) > FIELD_HALF_MM or abs(y) > FIELD_HALF_MM:
        raise ValueError(f"field ({x},{y}) outside +-{FIELD_HALF_MM}mm")
    fluence = fluence_Jcm2(energy_j, spot_um)
    shot = {
        "energy_j": float(energy_j),
        "spot_um": float(spot_um),
        "x_mm": x,
        "y_mm": y,
        "fluence_Jcm2": fluence,
        "simulated": True,
    }
    SHOT_LOG.append(shot)
    return shot


def clear_log():
    SHOT_LOG.clear()
    return 0
