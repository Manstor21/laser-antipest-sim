"""PR2 — Checklist IEC 60825-1 Clase 1M SIMULATED (sim-only, 0 EUR).

Rama atenuada <1.8uJ GREEN; nominal 3mJ RED by design (divergencia Class-4).
Composicion-only sobre `pulse_energy_j/DutyMeter/FireController`,
`fluence_Jcm2`, `r6_interlock` y `fire_authorize`. No muta codigo B1-5.

Etiquetado obligatorio: `1M+SIMULATED` + `NO-CERT (SIMULATED, sin certificar)`.
"""
import math

from jetson.fusion.fusion_pipeline import fire_authorize
from jetson.fusion.hard_rules import r6_interlock
from jetson.laser.fire_controller import (
    BURST_MAX,
    COOLDOWN_S,
    DUTY_LIMIT,
    DutyMeter,
    FireController,
    pulse_energy_j,
)
from sim.gazebo.f_theta_optics import fluence_Jcm2

SIMULATED = True
LABEL_1M = "1M+SIMULATED"
LABEL_NO_CERT = "NO-CERT (SIMULATED, sin certificar)"
E_LIMIT_UJ = 1.8  # frontera Clase 1M SIMULATED para este harness

# Rama atenuada: 1.0 W x 1.0 us = 1.0 uJ (< 1.8 uJ). Filtro ND ~1000x
# frente al nominal; dentro de WIDTH 1-5 us del FireController.
ATT_PEAK_W = 1.0
ATT_WIDTH_US = 1.0
# Rama nominal: 1000 W x 3.0 us = 3.0 mJ = 3000 uJ (Class-4 sim).
NOM_PEAK_W = 1000.0
NOM_WIDTH_US = 3.0

# MPE SIMULATED (supuesto de harness, NO certificacion): irradiancia de
# referencia para NOHD en SI. Solo ordena atenuada << nominal.
MPE_SIMULATED_J_M2 = 20.0
MPE_NOTE = "SIMULATED assumption, NO-CERT"


def _attenuated_energy_uJ() -> float:
    return pulse_energy_j(ATT_PEAK_W, ATT_WIDTH_US) * 1e6


def _nominal_energy_uJ() -> float:
    return pulse_energy_j(NOM_PEAK_W, NOM_WIDTH_US) * 1e6


def _nominal_verdict() -> str:
    return "RED-by-design"


def _nohd_m(energy_j: float) -> float:
    """NOHD SIMULATED: sqrt(4*E / (pi*MPE)). E en J, MPE en J/m2 -> m."""
    return math.sqrt(4.0 * float(energy_j) / (math.pi * MPE_SIMULATED_J_M2))


def _divergence_note() -> str:
    return (
        "Divergencia SIMULATED: nominal 3mJ = Class-4 sim, nunca 1M; "
        "solo la rama atenuada <1.8uJ es 1M+SIMULATED. " + LABEL_NO_CERT
    )


def _labels() -> str:
    return f"{LABEL_1M} SIMULATED {LABEL_NO_CERT}"


def _promoted_out():
    return {"promoted": True, "ai_vote": True,
            "gates": {"R1": True, "R2": True, "R3": True, "R4": True}}


def test_iec_attenuated_energy_below_1_8uJ():
    e_uj = _attenuated_energy_uJ()
    assert e_uj < E_LIMIT_UJ, f"E={e_uj} uJ >= {E_LIMIT_UJ} uJ"
    assert e_uj == 1.0, f"E atenuada={e_uj} uJ != 1.0 uJ"
    flu = fluence_Jcm2(e_uj * 1e-6, 500.0)
    assert math.isfinite(flu) and flu > 0


def test_iec_duty_within_0_1pct():
    dm = DutyMeter(limit=DUTY_LIMIT, window_s=1.0)
    t0 = 100.0
    for i in range(5):  # rafaga realista: 5 x 1 us en 10 ms
        assert dm.request(t=t0 + i * 0.002, width_us=ATT_WIDTH_US) is True
    assert dm.duty(t0 + 0.010) <= DUTY_LIMIT, "duty debe respetar 0.1% (1 s)"


def test_iec_burst_5_plus_cooldown_2s():
    fc = FireController(peak_w=ATT_PEAK_W)
    fc.try_arm(authorized=True, t=20.0)
    fc.start_firing(settle_ok=True, t=20.001)
    t = 20.002
    for i in range(BURST_MAX):
        ok, energy = fc.request_pulse(t=t + i * 0.001, width_us=ATT_WIDTH_US)
        assert ok is True, f"pulso {i} denegado en rafaga atenuada"
        assert energy * 1e6 < E_LIMIT_UJ
    ok, _ = fc.request_pulse(t=t + 0.006, width_us=ATT_WIDTH_US)
    assert ok is False, "pulso 6 debe denegarse (burst<=5)"
    assert fc.request_pulse(t=t + 0.500, width_us=ATT_WIDTH_US)[0] is False
    ok, _ = fc.request_pulse(t=t + COOLDOWN_S + 0.100, width_us=ATT_WIDTH_US)
    assert ok is True, "tras cooldown 2 s debe permitir de nuevo"


def test_iec_nohd_computed_and_nominal_diverges():
    att_m = _nohd_m(_attenuated_energy_uJ() * 1e-6)
    nom_m = _nohd_m(_nominal_energy_uJ() * 1e-6)
    assert math.isfinite(att_m) and att_m >= 0
    assert math.isfinite(nom_m) and nom_m > att_m, \
        f"NOHD nominal={nom_m} debe exceder atenuada={att_m}"
    assert MPE_NOTE.startswith("SIMULATED")


def test_iec_veto_r6_env_shutter_denies():
    assert r6_interlock(human_min_m=0.5, track_mm=10.0) is False  # humano <2 m
    assert r6_interlock(human_min_m=5.0, track_mm=50.0) is False  # track >40 mm
    assert r6_interlock(human_min_m=5.0, track_mm=10.0) is True
    out = _promoted_out()
    assert fire_authorize(out, False, True) is False, "veto entorno deniega"
    assert fire_authorize(out, True, False) is False, "shutter deniega"
    assert fire_authorize(out, True, True) is True, "todo claro autoriza"


def test_iec_label_1M_simulated_no_cert():
    labels = _labels()
    assert LABEL_1M in labels and "SIMULATED" in labels and "NO-CERT" in labels
    assert SIMULATED is True


def test_iec_nominal_3mJ_red_by_design():
    e_uj = _nominal_energy_uJ()
    assert e_uj == 3000.0, f"E nominal={e_uj} uJ != 3000 uJ"
    assert e_uj >= E_LIMIT_UJ, "nominal debe exceder el limite 1M"
    assert _nominal_verdict() == "RED-by-design"
    note = _divergence_note()
    assert "Class-4" in note and "NO-CERT" in note and "SIMULATED" in note
