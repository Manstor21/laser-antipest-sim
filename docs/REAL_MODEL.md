## Modelo real GBIF (lenguaje llano)

Que es: una regresion logistica (la herramienta estadistica mas simple para
decidir entre dos clases) con 35 numeros por foto. No es una red neuronal:
traza una sola linea recta de decision. Es el primer modelo ENTRENADO DE
VERDAD del proyecto (antes solo habia un muñeco de prueba con numeros fijos).

Como se entreno (`sim/dataset/train_real.py`):

- Fotos: las 380 de `data/real/gbif_velutina` (150) + `gbif_abeja` (150) +
  `gbif_crabro` (80). De cada foto se recorta el CENTRO 320x320, igual que
  en `replay_gbif.py`.
- Por foto se calculan 35 numeros rapidos: histograma de color H(16)+S(8)+V(8)
  normalizado, proporcion de pixeles oscuros (V<60) y claros (V>200), y
  densidad de bordes (Sobel). Se estandarizan (se guardan media y escala).
- Etiquetas: velutina=1, abeja/crabro=0. Reparto 80/20 estratificado con
  semilla 42: 304 para entrenar, 76 para validar.
- `LogisticRegression(max_iter=2000)`. Tardo ~0.2 min en CPU.
- Pesos exportados a `jetson/ai/real_model.npz`; la inferencia
  (`jetson/ai/real_backend.py`) usa solo numpy (sigmoide a mano), sin sklearn.

Metricas en validacion (76 fotos: 30 velutina, 27 abeja, 19 crabro):

- A umbral 0.5: recall 0.700 (21/30), FP abeja 7/27, FP crabro 4/19,
  precision 0.656.
- Umbral con cero falsas alarmas (FP_total=0): 0.990, con recall 0.133 (4/30).
  Filosofia duda->abeja: a ese umbral ninguna abeja ni crabro de validacion
  se confunde con velutina, pero se escapan la mayoria de velutinas.

Como reentrenar:

```text
python sim/dataset/train_real.py            # todo (380 fotos)
python sim/dataset/train_real.py --limit 60 # prueba rapida (60 por clase)
python -m pytest tests/test_real_backend.py -q
```

Limites honestos:

- El recorte es el CENTRO de la foto, sin cajas: el bicho puede salir
  cortado o fuera del recorte. Eso mete ruido y lastra el recall.
- 380 fotos son pocas; el modelo lineal solo mira color y bordes globales,
  no formas (no ve torax, bandas ni alas).
- NO apto para disparo real. El gate 0.995/0.001 de `thresholds.yaml` sigue
  mandando, y este modelo es diagnostico, sin valor de certificacion.

## YOLOv8n-cls de Colab (diagnostico, NO activo por defecto)

Que es: `DESCARGAS/best.pt` (YOLOv8n-cls, 2,8 MB, 1.442.131 parametros,
clases abeja/crabro/velutina), entrenado en Colab por el usuario y evaluado
en local con `sim/dataset/eval_yolo.py` (crop central -> 224, top1 por foto):

```text
python sim/dataset/eval_yolo.py            # todo (380 fotos)
python sim/dataset/eval_yolo.py --limit 10 # prueba rapida
```

Metricas honestas (16-sep-2026):

- Colab, split validacion 76 fotos (30 velutina, 30 abeja, 16 crabro):
  top1 global 0.855, recall velutina 0.867 (26/30), FP abeja->velutina 0/30,
  FP crabro->velutina 2/16, top1 abeja 0.967, crabro 0.625.
- Local, las 380 fotos COMPLETAS (150/150/80, incluye train: sesgo optimista):
  top1 global 0.818 (311/380), recall velutina 0.907 (136/150),
  FP abeja->velutina 11/150, FP crabro->velutina 25/80,
  top1 abeja 0.913, crabro 0.475.
- Ojo al comparar: Colab mide solo validacion (76) y local mide todo (380),
  asi que no son el mismo examen; el parecido en abeja/velutina confirma que
  el peso viajo bien, pero el recall local esta inflado por evaluar tambien
  sobre fotos de entrenamiento.
- Logistico local (validacion 76 fotos): recall 0.700 (21/30) a umbral 0.5;
  con cero falsas alarmas (umbral 0.990) recall 0.133 (4/30).

Backend opcional: `jetson/ai/yolo_backend.py` (`load()` /
`predict_proba(crop320) -> (Pv, Pb=1-Pv)`, con Pv = prob. softmax de
velutina, resize interno a 224; peso en `jetson/ai/yolov8n_velutina.pt`).
El stub `detector.py` y `thresholds.yaml` NO se tocan: el YOLO es
diagnostico y el logistico/gate actual sigue mandando.

Como activarlo en el futuro (cuando toque, no ahora):

1. Calibrar un umbral de Pv alto sobre validacion (igual que se hizo con el
   logistico: buscar FP_total=0 y medir su recall), porque el top1 directo
   NO es un umbral de seguridad.
2. Conectar el backend en `detector.py` tras el gate de `thresholds.yaml`
   y revalidar con `pytest tests/ -q`.

Limites: el recorte sigue siendo el CENTRO (sin cajas); crabro flojea
(top1 local 0.475, 25/80 fugan a velutina en top1); sin umbral calibrado NO
apto para disparo real; sin valor de certificacion.

## YOLOv8s-cls v2 de Colab (diagnostico; MANDA sobre v1 en comparativa local)

Que es: `jetson/ai/yolov8s_velutina_v2.pt` (copia de `DESCARGAS/best.pt`,
10,3 MB en disco, 5.089.703 parametros, 7 clases:
abeja/bombus/crabro/eristalis/germanica/velutina/vespula). v1
(`jetson/ai/yolov8n_velutina.pt`, 2,9 MB, 1.442.131 params, 3 clases:
abeja/crabro/velutina) sigue en su sitio; `detector.py` y `thresholds.yaml`
NO se tocan: ambos YOLO son diagnostico y el gate actual sigue mandando.

Evaluacion local v2:

```text
python sim/dataset/eval_yolo.py --weights jetson/ai/yolov8s_velutina_v2.pt --limit 380
```

Colab v2 (validacion 310 fotos, 7 clases, `DESCARGAS/resultados_v2.txt`):

- Top1 global 0.8935; recall velutina 0.9375 (75/80).
- FP hacia velutina: abeja 1/80, crabro 1/40,
  vespula/germanica/bombus/eristalis 0.
- `operating_point.json` dice {umbral 0.999, recall 0.0, fp 0}: a 0.999 el
  recall DE COLAB colapsa a 0; el punto FP=0 util en local esta mas abajo
  (ver barrido).

Local 380 fotos (150 velutina / 150 abeja / 80 crabro, incluye train: sesgo
optimista), top1 directo:

- v2: top1 0.855 (325/380), recall velutina 0.947 (142/150),
  FP abeja->velutina 5/150, FP crabro->velutina 14/80,
  top1 abeja 0.913, top1 crabro 0.575 (el resto de fallos fugan a clases
extra: germanica/vespula/eristalis).
- v1: top1 0.818 (311/380), recall velutina 0.907 (136/150),
  FP abeja->velutina 11/150, FP crabro->velutina 25/80,
  top1 abeja 0.913, top1 crabro 0.475.
- v2 gana en las 4 casillas (top1, recall, FP abeja, FP crabro).

Barrido de umbral Pv>=U sobre v2 en local (FP = abeja+crabro promovidos por
error; a partir de 0.90 la abeja ya es 0 y el que manda es crabro):

- 0.90: recall 0.880 (132/150), FP 7 (abeja 0, crabro 7).
- 0.95: recall 0.860 (129/150), FP 5 (abeja 0, crabro 5).
- 0.98: recall 0.813 (122/150), FP 2 (abeja 0, crabro 2).
- 0.99: recall 0.767 (115/150), FP 1 (abeja 0, crabro 1).
- 0.995: recall 0.747 (112/150), FP 1 (abeja 0, crabro 1).
- 0.999: recall 0.633 (95/150), FP 0 (abeja 0, crabro 0).
- (El punto 0.90 se verifico tambien con `eval_yolo.py --umbral 0.90`; el
  resto sale de un one-liner que reutiliza sus funciones `glob_images` /
  `load_rgb` / `center_crop_224` con una sola carga del modelo, sin editar
  el fichero.)

Mejor umbral con FP_total=0, busqueda fina (umbral justo por encima del Pv
maximo de los 230 negativos):

- v2: U ~= 0.9958 -> recall 0.720 (108/150), FP 0/230.
- v1: U ~= 0.9994 -> recall 0.193 (29/150), FP 0/230.
- Al mismo umbral 0.9958: v2 FP 0/recall 0.72 frente a v1 FP 3/recall 0.42;
  a 0.999: v2 FP 0/recall 0.633 frente a v1 FP 1/recall 0.26.

Veredicto (16-sep-2026): MANDA v2. A FP=0 casi cuadruplica el recall de v1
(0.720 frente a 0.193) y en top1 directo tambien gana en todo (top1 0.855
frente a 0.818, recall 0.947 frente a 0.907, FP 5+14 frente a 11+25). Cuando
toque activarlo: calibrar el umbral sobre validacion pura y conectar tras el
gate; hasta entonces sigue siendo diagnostico.

Limites honestos:

- Colab-val (310 fotos, 7 clases) y local (380 fotos, 3 clases) no son el
  mismo examen; no se comparan literalmente.
- Local incluye fotos de train: top1/recall optimistas en AMBOS modelos.
- El recorte sigue siendo el CENTRO (sin cajas); crabro flojea en ambos
  (top1 local 0.575 v2 / 0.475 v1).
- El operating_point.json de Colab (recall 0.0 a 0.999) NO reproduce en
  local (v2 a 0.999 da recall 0.633): el umbral hay que calibrarlo aqui,
  no copiarlo del cuaderno.
- Nota: `jetson/ai/yolo_backend.py` sigue apuntando a v1 y su mensaje de
  error dice "solo si <10MB" (v2 pesa 10,3 MB); si v2 se promueve a
  backend habra que actualizar path y texto.
- Sin valor de certificacion.

## YOLOv8n-detector local (cajas) — Hornet3000+ 3000 fotos, entrenado 100% local CPU

Que es: `jetson/ai/yolov8n_det.pt` (6 MB, 3.0M params, YOLOv8n-detector, 3 clases:
vespa_velutina/vespa_crabro/vespula_vulgaris), entrenado 100% local en CPU
sin Kaggle/Colab. Peso es copia stripped de `runs/detect/train/weights/best.pt`
(original 23 MB con optimizador -> stripped 5.9 MB con `strip_optimizer`).

Dataset: `data/real/hornet3000/data3000/data/{train,val}/{images,labels}/Vespa_*`
(3088 train + 297 val), aplanado a `data/det/train|val/{images,labels}/` por
`sim/dataset/prepare_hornet.py` (subcarpetas Vespa_velutina etc -> plano;
`data/det/data.yaml` con `path` absoluto, `names: [vespa_velutina, vespa_crabro, vespula_vulgaris]`).
Train plano: 3088 imágenes (2671 con label, 417 fondo sin label), val 297/297;
huérfanos omitidos (157 txt train sin imagen, 3 txt val sin imagen en val pero imagen en train).
Clases YOLO 0=velutina 1=crabro 2=vulgaris (igual que config.yaml).

Como se entrenó (`sim/dataset/train_det.py`):

```text
python sim/dataset/prepare_hornet.py
python sim/dataset/train_det.py  # YOLO("yolov8n.pt").train(data=data/det/data.yaml, epochs=10, imgsz=320, batch=8, device=cpu, workers=2, patience=3)
python sim/dataset/eval_det.py
```

- Base `yolov8n.pt` (6.2 MB COCO). 5 épocas efectivas (intento 10, timebox 30 min;
  1.4 it/s en CPU -> 25 min para 5 épocas, 6ª interrumpida por timeout y descartada).
  early stopping `patience=3`, `project=runs/detect` `name=train`.
- `runs/detect/train/weights/best.pt` 23 MB original; `best_stripped.pt` 5.9 MB copiado a `jetson/ai/yolov8n_det.pt`.
  No pisa `yolov8n_velutina.pt` ni `yolov8s_velutina_v2.pt` (cls siguen).
- Backend `jetson/ai/det_backend.py` con `load()/detect(image)->[boxes]` (ultralytics,
  error claro si no hay pesos, CPU, conf 0.25, iou 0.45, imgsz 320). `detector.py` y
  `yolo_backend.py` NO tocados: el detector de cajas es diagnóstico y el stub/gate sigue mandando.

Métricas (17-sep-2026, 5 épocas, val 297 imgs, data/det):

- `model.val()` (IoU 0.5:0.95 COCO): precision 0.845, recall 0.735, mAP50 0.841, mAP50-95 0.593.
  Por clase AP50: velutina 0.838, crabro 0.812, vulgaris 0.873 (bien balanceado).
  `results.csv` época 5: box_loss 1.34, cls_loss 1.49, dfl 1.35; val box 0.97, cls 1.06.
- `model.predict` custom @conf 0.5 IoU 0.5: GT velutina 119 cajas en val,
  TP 38, recall velutina 0.319 (38/119), FP duros 1 caja velutina en 198 imgs sin velutina
  (0.005 FP/img, total pred velutina 40). @conf 0.25 el recall sube pero FP también
  (mAP alto porque integra todos los umbrales).
- Guardado en `resultados_det_local.txt` y `jetson/ai/box_thresholds_det.json`
  (umbral 0.5, recall 0.319, fp 1, mAP 0.841).

Veredicto honesto:

- DEMUESTRA: detector REAL entrenado de verdad (localiza cajas, no solo clasifica crop).
  Aprende en 5 épocas: mAP50 0.841 >> aleatorio; FP duros 1 muy bajo a 0.5 indica buena separación.
- LÍMITES: solo 5 épocas por CPU lenta (no 10); val pequeño (297 imgs, 119 velutina) -> mAP optimista;
  417 fondos sin label en train y desbalance Vespula (797 vs 1186/1072) + sin augmenting fuerte;
  recall 0.319 a 0.5 es bajo para disparo (muchos velutina no se detectan a conf 0.5); necesita
  calibrar umbral por clase y más épocas / datos / augment.
- NO apto para disparo real; diagnóstico sin certificación. El gate `thresholds.yaml` sigue mandando.

Como reproducir:

```text
python sim/dataset/prepare_hornet.py --check
python sim/dataset/eval_det.py   # re-evalúa best.pt
pytest tests/test_det_train.py -q
```
