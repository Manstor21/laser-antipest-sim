"""Hard-gate chain R1-R4 (blocking) + R5 advisory (sim-only).

Doubt defaults to bee (no-fire): every gate must pass for promotion.
Thresholds mirror jetson/fusion/config.yaml; R5 only boosts confidence.
"""

R1_MIN, R1_MAX = 15.0, 35.0
R2_HOVER_MAX, R2_TRANSIT_MIN, R2_TRANSIT_MAX, R2_MIN_FRAMES = 0.5, 1.0, 11.0, 3
R3_THORAX_V_MAX, R3_BAND_H_MIN, R3_BAND_H_MAX, R3_BAND_S_MIN = 60, 15, 45, 80
R4_MIN, R4_MAX = 1.6, 2.4
R5_MIN, R5_MAX = 100.0, 170.0


def r1_size_ok(longest_axis_mm):
    """R1: 15-35 mm longest axis; <12 / >40 and doubt band reject."""
    v = float(longest_axis_mm)
    return R1_MIN <= v <= R1_MAX


def r2_classify(velocity_hist):
    """R2: hover (|v|<=0.5) vs transit (1-11 m/s), sustained >=3 frames."""
    hist = [abs(float(v)) for v in velocity_hist]
    if len(hist) < R2_MIN_FRAMES:
        return "invalid"
    if max(hist) <= R2_HOVER_MAX:
        return "hover"
    if all(R2_TRANSIT_MIN <= v <= R2_TRANSIT_MAX for v in hist):
        return "transit"
    return "invalid"


def r3_hsv_ok(thorax_v, band_h, band_s):
    """R3: dark thorax (V<60) + orange band (H15-45, S>80)."""
    return float(thorax_v) < R3_THORAX_V_MAX and \
        R3_BAND_H_MIN <= float(band_h) <= R3_BAND_H_MAX and \
        float(band_s) > R3_BAND_S_MIN


def r4_ratio_ok(wingspan, body):
    """R4: wingspan/body morph ratio 1.6-2.4."""
    body = float(body)
    if body <= 0:
        return False
    return R4_MIN <= float(wingspan) / body <= R4_MAX


def r5_score(sideband_hz):
    """R5 advisory: 100-170 Hz micro-Doppler boosts priority, never gates."""
    try:
        f = abs(float(sideband_hz))
    except (TypeError, ValueError):
        return 0.0
    return 0.3 if R5_MIN <= f <= R5_MAX else 0.0


# R6 blocking proximity interlock (SIMULATED, sim-only 0€).
R6_HUMAN_MIN_M = 2.0
R6_TRACK_MAX_MM = 40.0


def r6_interlock(human_min_m, track_mm):
    """R6: inhibit if human/animal <2m or any track >40mm.

    Returns True when clear (may fire), False when inhibited.
    Missing/stale track (None) defaults to inhibited (False).
    """
    try:
        track = float(track_mm)
    except (TypeError, ValueError):
        return False
    if track > R6_TRACK_MAX_MM:
        return False
    if human_min_m is None:
        return True
    try:
        human = float(human_min_m)
    except (TypeError, ValueError):
        return False
    if human < R6_HUMAN_MIN_M:
        return False
    return True


def gate_chain(features):
    """Run R1-R4; promote only if ALL pass. Returns verdict dict."""
    gates = {
        "R1": r1_size_ok(features["longest_axis_mm"]),
        "R2": r2_classify(features["velocity_hist"]) in ("hover", "transit"),
        "R3": r3_hsv_ok(features["thorax_v"], features["band_h"],
                         features["band_s"]),
        "R4": r4_ratio_ok(features["wingspan"], features["body"]),
    }
    bonus = r5_score(features.get("sideband_hz", 0.0))
    passed = sum(gates.values())
    if all(gates.values()):
        confidence = min(0.6 + 0.1 * passed / 4 + bonus, 0.99)
        reason = "all hard gates pass"
        promoted = True
    else:
        failed = sorted(k for k, v in gates.items() if not v)
        confidence = 0.1
        reason = f"bee-safe reject: {','.join(failed)} fail (doubt->bee)"
        promoted = False
    return {"promoted": promoted, "gates": gates, "confidence": confidence,
            "reason": reason, "r5_bonus": bonus}
