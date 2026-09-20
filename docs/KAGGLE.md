# Velutina — Entrenar detector YOLO en Kaggle manos libres (0 euros)

Guia en espanol llano. Todo pasa en Kaggle, sin `kaggle.json` ni descargas en tu PC. Una vez le das a Run, puedes **apagar el PC tranquilo**: sigue entrenando en Kaggle.

## Que vas a conseguir

Un detector YOLO (`yolov8n.pt`, ~6 MB) entrenado con Hornet3000 (cajas YOLO, 3 clases: velutina/crabro/vulgaris). Solo descargaras al final el `best.pt` + 2 ficheros txt/json (~6 MB).

## Pasos Kaggle clic a clic

1. **Ir al dataset**: abre https://www.kaggle.com/datasets/marcoryvandijk/vespa-velutina-v-crabro-vespulina-vulgaris (dataset Hornet3000 `vespa-velutina-v-crabro-vespulina-vulgaris`). Si no tienes cuenta, crea una gratis (vale con Google).

2. **New Notebook**: en la pagina del dataset clica **"New Notebook"** (o "+ New Notebook"). Se abre un notebook vacio en https://www.kaggle.com/code.

3. **Add data**: en el panel derecho del notebook clica **"Add data"** -> busca `vespa-velutina-v-crabro-vespulina-vulgaris` -> **Add** (el dataset aparecera en `/kaggle/input/vespa-velutina-v-crabro-vespulina-vulgaris/`). Verifica que lo ves en Input a la derecha.

4. **GPU T4 x2**: arriba a la derecha clica **Settings** -> **Accelerator** -> elige **GPU T4 x2** (gratis). Deja el resto por defecto. Sin GPU tambien funciona pero tarda horas en CPU.

5. **Importar cuaderno y Run All**: en el notebook vacio clica **File -> Import Notebook** y sube `notebooks/train_yolo_detect_kaggle_handsfree.ipynb` de este repo (o copia-pega las celdas). Luego clica **Run All** (o ejecuta las celdas en orden). Tarda ~1h (50 epocas, imgsz 640, batch 16). **Puedes apagar el PC tranquilo**: el entreno sigue en los servidores de Kaggle. Cuota gratis: 30h de GPU por semana, max 9h por sesion.

6. **Output -> Download**: al terminar, en el panel derecho clica **Output**. Veras:
   - `/kaggle/working/runs/det/weights/best.pt` (~6 MB)
   - `/kaggle/working/resultados_det_kaggle.txt`
   - `/kaggle/working/box_thresholds_kaggle.json`
   Descarga `best.pt` (click derecho -> Download) y los dos txt/json. En tu repo guarda `best.pt` como `jetson/ai/yolov8n_det_kaggle.pt` y los txt/json junto al peso para comparar.

## Tiempo y cuota

- ~1h con GPU T4 x2 (50 epocas). Sin GPU no compensa.
- Cuota Kaggle gratis: 30h/semana de GPU, 9h por sesion. Este entreno gasta ~1h.
- El dataset y los logs se quedan en Kaggle; en local solo quedan ~6 MB.

## Que significan los numeros

- `mAP50` / `mAP50-95`: calidad de cajas (cuanto mejor, mas clava la caja y la clase).
- `recall velutina @0.5`: de cada 100 velutinas, cuantas pilla con conf 0.5.
- `FP duros`: cajas "velutina" en imagenes sin velutina (crabro/vulgaris/fondo). Si es alto, dispara a bichos buenos.

## Limite honesto

Fotos de observadores != vuelo real en tu apiario. El `resultados_det_kaggle.txt` no vale para campo: valida con fotos de tu piquera antes de habilitar disparo.
