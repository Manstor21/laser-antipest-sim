# Impacto en fauna SIMULATED (sim-only, 0 EUR)

> **SIMULATED sin certificar (NO-CERT): bycatch==0 por construccion (FP=0).**
> Matriz calculada en `tests/test_selectivity_fauna.py` (5/5) sobre la jaula
> N=200 dual-seed. Sin campo ni observacion real.

## Matriz especies x efecto

| Especie | n | promoted | shot | Efecto |
|---------|---|----------|------|--------|
| vespa_velutina (objetivo) | 40 | 40 | 40 | target-suppression SIMULATED |
| apis_mellifera (no-objetivo) | 160 | 0 | 0 | no-harm SIMULATED |
| **bycatch** | — | **0** | — | 0 por construccion (bee_FP==0 del gate) |

## Acotacion de exposicion

- Veto R6 + entorno + shutter via `fire_authorize()`: humano<2 m, track>40 mm,
  env/shutter falsos => denegado y logueado (3/3 casos veto deniegan).
- Caps duty<=0.1 % (`DutyMeter`) y burst<=5+2 s: exposiciones == promovidos
  velutina (<=40), bycatch 0.
- Adverso 4x12 y gate 4-cond (recall>0.90, bee_FP==0, prec>=0.95, IoU>=0.80)
  sostienen el cero-by-catch en simulacion.

## Limites

Sim-only: sin mortalidad real, sin deriva, sin sub-especies. Cualquier uso de
campo requeriria estudio ecologico y autorizacion; este doc no certifica nada.
Rollback: borrar test + doc; src/goldens intactos.
