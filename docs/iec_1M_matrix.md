# Matriz IEC 60825-1 Clase 1M SIMULATED (sim-only, 0 EUR)

> **1M+SIMULATED — NO-CERT (SIMULATED, sin certificar).** Checklist calculado
> en `tests/test_selectivity_iec.py` (7/7). No es certificacion ni ensayo de
> laboratorio. MPE y NOHD son supuestos de harness.

## Limite y ramas

| Rama | Energia | Veredicto | Nota |
|------|---------|-----------|------|
| Atenuada (1.0 W x 1.0 us) | 1.0 uJ < 1.8 uJ | GREEN | Filtro ND ~1000x; etiqueta `1M+SIMULATED` |
| Nominal (1000 W x 3.0 us) | 3000 uJ = 3 mJ | RED-by-design | Class-4 sim; divergencia declarada, nunca 1M |

## Checklist (rama atenuada)

1. **Energia:** `pulse_energy_j(1.0,1.0)` x1e6 = 1.0 uJ < 1.8 uJ (+`fluence_Jcm2` finita).
2. **Duty:** 5 x 1 us en 10 ms via `DutyMeter` <= 0.1 % (ventana 1 s).
3. **Burst:** `FireController` <= 5 pulsos; pulso 6 denegado; cooldown 2 s
   (`BURST_MAX`, `COOLDOWN_S`).
4. **NOHD SIMULATED:** `sqrt(4*E/(pi*MPE))`, MPE=20.0 J/m2 (supuesto NO-CERT);
   solo ordena nominal >> atenuada.
5. **Vetos:** R6 (`human>=2 m`, `track<=40 mm`) + `fire_authorize(env,shutter)`;
   cualquier veto deniega.
6. **Etiquetado:** `1M+SIMULATED SIMULATED NO-CERT (SIMULATED, sin certificar)`.

## Divergencia nominal

Nominal 3 mJ excede el limite 1M por 1666x: resultado RED-by-design con nota
"nominal 3mJ = Class-4 sim, nunca 1M; solo la rama atenuada <1.8uJ es
1M+SIMULATED. NO-CERT". Sin hardware no hay emision real.
