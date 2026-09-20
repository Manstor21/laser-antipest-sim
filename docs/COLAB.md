# Velutina — Entrenar YOLO real con 0 MB en local (todo en Colab gratis)

Esta guía es en español llano. No necesitas saber programar: solo seguir pasos.

## Qué vas a conseguir

Un modelo YOLO de verdad (clasificación `yolov8n-cls`), entrenado con ~380 fotos
reales de GBIF, sin usar casi disco en tu ordenador. Lo único que descargarás al
final pesa **~6 MB** (el fichero `best.pt` + `resultados.txt`).

En local no se usa casi disco (solo ~6MB del .pt). Todo lo pesado (dataset,
entrenamiento, GPU) ocurre en Google Colab, que es gratis.

## Paso 1 — Abrir Colab (cuenta Google gratis)

1. Entra en https://colab.google.com
2. Si te pide iniciar sesión, usa cualquier cuenta Google gratis.
3. No hace falta pagar nada ni meter tarjeta.

## Paso 2 — Subir el cuaderno

1. En Colab: `Archivo → Subir cuaderno`.
2. Sube el fichero `notebooks/train_yolo_colab.ipynb` de este repo.
3. Verás las celdas en orden: GPU, descarga GBIF, split, entreno, evaluación,
   export ONNX y conclusiones.

## Paso 3 — Entorno con GPU (gratis, celdas en orden)

1. En Colab: `Entorno de ejecución → Cambiar tipo de entorno → GPU gratis`
   (si no hay GPU disponible, el cuaderno sigue funcionando en CPU con 3 epochs).
2. Ejecuta las celdas en orden, o usa `Entorno de ejecución → Ejecutar todo`.
3. No te saltes el orden: cada celda es idempotente (se puede re-ejecutar sin
   romper nada) e imprime su progreso.

## Paso 4 — Ejecutar todo (~15-30 min)

- Con GPU gratis tarda unos 15-30 min (descarga GBIF + 25 epochs).
- Sin GPU (CPU) tarda parecido pero solo hace 3 epochs de prueba.
- No necesitas claves ni cuentas en ningún sitio: GBIF es API pública.

## Paso 5 — Descargar best.pt + resultados.txt

1. Al terminar, en el panel izquierdo de Colab abre `Archivos`.
2. Busca `runs/classify/train/weights/best.pt` (~5MB) y `resultados.txt`.
3. Click derecho → Descargar cada uno.
4. En tu repo ponlos aquí:
   - `best.pt` → `jetson/ai/yolov8n_velutina.pt` (sobrescribe/copia con ese nombre)
   - `resultados.txt` → guárdalo donde quieras para comparar (p. ej. junto al .pt)
5. Lo único que descargará ocupa ~6MB en total. El dataset (~380 fotos) se queda
   en Colab y no ocupa nada en tu disco local.

## Qué significan los números (resultados.txt)

- `top1`: precisión global (1 de cada cuántas fotos acierta a la primera).
- `top1 por clase`: precisión separada para velutina / abeja / crabro.
- `recall velutina`: de cada 100 velutinas reales, cuántas detecta. Es lo más
  importante (queremos no dejar escapar velutinas).
- `FP abeja/crabro`: cuántas abejas o avispa europea se confunden con velutina
  (falsos positivos). Si es alto, el sistema ahuyentaría o marcaría bichos buenos.
- Matriz de confusión (`confusion_matrix.png` en Colab): tabla de
  "realidad vs predicción". La diagonal buena = aciertos.

Ejemplo: recall velutina 0.85 = de 100 velutinas pilla 85 y se le escapan 15.
FP abeja 10 = 10 abejas marcadas como velutina por error.

## Límites honestos

- Son ~380 fotos centradas de museo/GBIF, no avispas en vuelo real ni con tu cámara.
- Es clasificación (¿qué bicho es esta foto recortada?), no detección+localización
  (¿dónde hay bichos en una imagen con fondo?). GBIF no trae cajas, por eso.
- Con tan pocas fotos el modelo puede fallar con fondos, luz o poses nuevas.
- Lo que demuestra: que el pipeline funciona de punta a punta con datos reales y
  licencias limpias. No demuestra que funcione ya en campo.

## Siguiente paso (detección de verdad)

Para detectar + localizar en vuelo necesitas un dataset con cajas
(Kaggle / Roboflow con etiquetas de bounding box) y entrenar `yolov8n.pt`
detector, no `yolov8n-cls.pt`. Ese será el siguiente cuaderno.

---

## Cuaderno v2 (más precisión, misma idea: todo en Colab gratis)

Lo anterior sigue valiendo tal cual para v1. El cuaderno
`notebooks/train_yolo_colab_v2.ipynb` es una versión aparte
(`train_yolo_colab.ipynb` queda intacto) que sube la precisión sin cuentas
ni disco local.

### Qué cambia v2 vs v1

1. Modelo mayor: `yolov8s-cls.pt` en vez de `yolov8n-cls.pt` (~12MB el best).
2. Más fotos GBIF: velutina 400 + abeja 400 + crabro 200 (v1: 150/150/80).
3. Negativos nuevos parecidos: Vespula vulgaris 200, Vespula germanica 150,
   Bombus terrestris 100 y Eristalis tenax 100 (mosca que imita abeja).
4. Aumento suave en el entreno
   (`degrees=15, translate=0.1, scale=0.2, fliplr=0.5, hsv_h=0.02, hsv_s=0.4`)
   con `EPOCHS=50` (3 si CPU), `IMGSZ=320`, `BATCH=64`.
5. Celda clave de calibración FP=0: barrea el umbral Pv 0.5→0.999 sobre val y
   guarda `operating_point.json` {umbral, recall, fp}: a qué umbral no toca
   ninguna abeja y cuántas velutinas pilla.

### 5 pasos (v2)

1. En https://colab.google.com: `Archivo → Subir cuaderno` y sube
   `notebooks/train_yolo_colab_v2.ipynb`.
2. `Entorno de ejecución → Cambiar tipo de entorno → GPU gratis`
   (sin GPU funciona con 3 epochs de prueba).
3. `Entorno de ejecución → Ejecutar todo` (~30-60 min con GPU).
4. Descarga del panel Archivos: `best.pt` (~12MB) + `resultados_v2.txt` +
   `operating_point.json`.
5. En tu repo copia `best.pt` como `jetson/ai/yolov8n_velutina_v2.pt`
   (nombre distinto: así v1 y v2 conviven) y guarda los dos `.txt`/`.json`
   junto al peso.

### Cómo comparar con v1

```bash
python sim/dataset/eval_yolo.py --weights jetson/ai/yolov8n_velutina_v2.pt --umbral 0.99
```

Sin `--umbral` el informe es idéntico al de siempre (top1 + matriz).
Con `--umbral` (usa el de tu `operating_point.json`) aplica Pv>=umbral como
promote y reporta recall/FP a ese umbral con tus fotos locales de
`data/real/gbif_*`. Compara su `resultados_v2.txt` con el `resultados.txt`
de v1: más recall de velutina con cero FP es mejor.

La sección opcional del final solo se activa si subes tu `kaggle.json`
(descarga Hornet3000+, entrena detector `yolov8n.pt` 30 epochs con sus cajas
YOLO y compara cls vs det en 5 líneas); sin él la celda salta con mensaje,
sin error.

---

## Detector con cajas (Kaggle)

Cuaderno aparte: `notebooks/train_yolo_detect_kaggle.ipynb`. Localiza avispas
en la escena con cajas (clases 0=velutina, 1=crabro, 2=vulgaris) usando el
dataset Hornet3000+ de Kaggle
(https://www.kaggle.com/datasets/marcoryvandijk/vespa-velutina-v-crabro-vespulina-vulgaris,
guía https://github.com/vespCV/hornet3000). Necesita tu `kaggle.json`
(el cuaderno no trae ninguna clave).

6 pasos del usuario (clic a clic):

1. Crea tu cuenta gratis en https://www.kaggle.com (entra con Google si quieres).
2. Consigue tu token: avatar (arriba derecha) → Settings → baja a API →
   Create New Token. Se descarga `kaggle.json` a tu ordenador.
3. En https://colab.google.com: `Archivo → Subir cuaderno` y sube
   `notebooks/train_yolo_detect_kaggle.ipynb`. Luego abre el panel Archivos
   (carpeta a la izquierda) y arrastra tu `kaggle.json` dentro
   (debe quedar en `/content/kaggle.json`).
4. Activa GPU gratis: `Entorno de ejecución → Cambiar tipo de entorno → GPU`.
   (Sin GPU funciona con 5 epochs de prueba.)
5. `Entorno de ejecución → Ejecutar todo` (~1 h con GPU: descarga + 30 epochs
   `yolov8n.pt` imgsz 640 batch 16 + evaluación + export ONNX). Si falta el
   `kaggle.json`, la celda 2 lo dice claro y para sin error.
6. Descarga 2 ficheros del panel Archivos (~6 MB en total):
   `runs/detect/train/weights/best.pt` y `resultados_det.txt`. En tu repo
   guarda el peso como `jetson/ai/yolov8n_det.pt` (nombre distinto: NO pisa
   los cls `yolov8n_velutina.pt` / `yolov8n_velutina_v2.pt`) y el txt junto a él.
