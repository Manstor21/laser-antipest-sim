# Datasets reales para el replay sim-to-real (paso a paso)

Guía en lenguaje llano para descargar avispas y abejas reales y pasarlas por
`sim/dataset/replay_real.py`. El informe que sale es solo un
**DIAGNÓSTICO SIM-TO-REAL, sin valor de certificación**.

## 1) Hornet3000+ (Kaggle)

1. Crea una cuenta gratis en https://www.kaggle.com (Sign Up, confirma el correo).
2. Abre el dataset:
   https://www.kaggle.com/datasets/marcoryvandijk/vespa-velutina-v-crabro-vespulina-vulgaris
3. Pulsa el botón **Download** (son cientos de MB, tarda unos minutos).
4. Descomprime el zip dentro de `data/real/hornet3000/` para que quede así:
   `data/real/hornet3000/data.yaml` + `images/` + `labels/`.

## 2) Vespa velutina 3715 (Roboflow)

1. Crea una cuenta gratis en https://universe.roboflow.com (Sign Up).
2. Abre el proyecto:
   https://universe.roboflow.com/vespa-velutina-pobjr/vespa-velutina-nubcn/dataset/1
3. Pulsa **Download**, elige formato **YOLOv8** (es el mismo layout YOLO:
   `data.yaml` + `images/` + `labels/`).
4. Descomprime en `data/real/velutina3715/`.

## 3) Abejas Honey Bee 909 (negativos, Roboflow)

1. Con la misma cuenta gratis de Roboflow, abre:
   https://universe.roboflow.com/bscs-kxc9w/honey-bee-detection-model-zgjnb-8fmzo/dataset/1
2. Pulsa **Download**, formato **YOLOv8**.
3. Descomprime en `data/real/abejas/`.

## Estructura esperada de carpetas

```text
data/real/
  hornet3000/
    data.yaml        # con `names:` y rutas train/val
    images/
      *.jpg
    labels/
      *.txt          # cada línea: cls cx cy w h (normalizados)
  velutina3715/
    data.yaml
    images/
    labels/
  abejas/
    data.yaml
    images/
    labels/
```

Si un dataset viene como `images/` + `labels/` + `classes.txt` (sin `data.yaml`),
también vale: el script lo detecta solo.

## Comandos

```bash
python sim/dataset/replay_real.py --data data/real/hornet3000 --limit 200
python sim/dataset/replay_real.py --data data/real/velutina3715 --limit 200
python sim/dataset/replay_real.py --data data/real/abejas --limit 200
```

- `--data DIR` es obligatorio; sin él (o con carpeta vacía) el script muestra
  esta misma ayuda y sale con código 2, sin traceback.
- `--limit N` (defecto 200): cuántas cajas evalúa como máximo.
- `--split val|train` (defecto `val`): qué partición del `data.yaml` usar.

## Cómo leer el informe

- **recall velutina**: de cada 100 velutinas reales, cuántas promociona el gate.
  `recall = TP / (TP + FN)`. Cerca de 1 = las pilla casi todas.
- **FP por clase**: cuántas avispas/abejas que NO son velutina fueron
  promocionadas por error, desglosado por clase (`crabro`, `apis`, …).
  Cerca de 0 = pocas alarmas falsas.
- **precisión**: de cada 100 promociones, cuántas eran velutina de verdad.
  `precisión = TP / (TP + FP)`.
- **Qué valores esperar del stub**: el stub calcula
  `Pv = 0.10 + 0.85*fg` y `Pb = 0.30*(1-fg) + 0.02*fg` donde `fg` es la
  fracción de píxeles brillantes (>100) del recorte. Con los umbrales
  `Pv>=0.995` y `Pb<=0.001` casi ningún recorte real promociona (haría falta
  `fg` ≈ 1.0, imagen casi toda blanca, y además pasar R3/R4 de color y forma).
  Por eso espera un **recall bajo y pocos FP**: es normal, el stub solo sirve
  para probar el cableado, no es el modelo final. Cuando cambies al modelo de
  verdad, el recall debería subir.
- El informe siempre termina con:
  **DIAGNÓSTICO SIM-TO-REAL, sin valor de certificación.**

## Licencias y atribución obligatoria

- Gran parte de estas fotos vienen de **Observation.org** y de subidas de
  usuarios: respeta la licencia de cada imagen/fichero (muchas son
  compartibles tipo **CC-BY**: úsalas dando crédito al autor).
- Revisa el `data.yaml`/README de cada descarga: si pone "no comercial",
  **no uses esos datos con fines comerciales sin revisar** la licencia antes.
- Atribución mínima al publicar resultados: nombra el dataset (Hornet3000+,
  velutina 3715, Honey Bee 909), su fuente (Kaggle / Roboflow / Observation.org)
  y la licencia aplicable.
