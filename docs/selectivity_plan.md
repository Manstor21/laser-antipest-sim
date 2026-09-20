# Plan de selectividad SIMULATED (sim-only, 0 EUR)

> **SIMULATED — NO-CERT (sin certificar).** Jaula virtual + Go/No-Go + IEC 1M +
> fauna + MLOps verificados solo por `tests/test_selectivity_*.py`. Sin campo,
> sin hardware, sin certificacion. Goldens y codigo B1-5 solo-lectura.

## 1. Jaula virtual bee-heavy (N=200)

- Composicion: `replay_scenario('bee_heavy',7)` (100) + `(...,107)` (100).
- Mezcla: 160 bees / 40 velutinas (bee_ratio 0.80), 5 frames/caso.
- Harness: `FusionPipeline.step_with_ai` x5 con Pv/Pb truth-mapped
  (velutina 0.997/0.0004, bee 0.10/0.40); `fire_authorize` para shot.
- Adverso: 4x12 `rain(11)/glare(7)/occlusion(7)/dusk(7)`; bees_promoted==0.
- Determinismo: igualdad seed7/seed11; `tests/goldens/bee_heavy.yaml` intacto.
- Verifier: `tests/test_selectivity_cage.py` (4/4).

## 2. Puerta Go/No-Go

- Go solo si recall_velutina>0.90 AND bee_FP==0 AND precision>=0.95
  AND IoU>=0.80 via `metrics.compute()` + `operating_point(0.90)`.
- Reporta `transfer_risk` + `recall_cost`; No-Go determinista si bee_FP>0.
- Verifier: `tests/test_selectivity_gonogo.py` (4/4, veredicto Go).

## 3. Seguridad laser + fauna (resumen)

- Rama atenuada 1.0 W x 1 us = 1.0 uJ < 1.8 uJ GREEN `1M+SIMULATED`;
  nominal 1000 W x 3 us = 3 mJ RED-by-design (Class-4 sim). Detalle en
  `docs/iec_1M_matrix.md`. Fauna bycatch==0 en `docs/fauna_impact.md`.

## 4. Trazabilidad MLOps

- `model_version`: ckpt_sha=sha256(`yolov8s-seg:COCO`), opset=12
  (`export_onnx.OPSET`), thresholds_hash=sha256(`jetson/ai/thresholds.yaml`),
  synth_ratio=0.357 (manifest 3000/8400=0.35714...).
- `run_log.jsonl`: 200 lineas {t,seed,truth,gates,Pv,Pb,vote,promoted,shot,
  version:{ckpt_sha,thresholds_hash,opset}}.
- Cambio de thresholds => hash distinto (detectado en test).
- Verifier: `tests/test_selectivity_mlops.py` (6/6).

## 5. Gate completo + rollback

- Full: `pytest tests/test_selectivity_*.py -v` (cage+gonogo+iec+fauna+mlops).
- Rollback: borrar 5 tests + 3 docs; src/goldens intactos.
