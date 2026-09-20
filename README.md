# laser-antipest-sim — el "Photon Matrix" para velutina, pero simulado y sin humo

> Un simulador completo del cacharro chino que caza mosquitos a láser, recalibrado para la velutina. Todo en Python, sin comprar nada, sin quemar nada. Si algún día quieres montarlo de verdad, aquí tienes los planos y las pruebas. Si no, te sirve como avisador barato para el colmenar.

Hecho porque estoy harto de ver colmenas reventadas en verano y de trampas que se llevan por delante a todo. La idea no es inventar otro cebo: es copiar la arquitectura del Photon Matrix (LiDAR + radar + cámara -> IA -> galvo -> láser) y cambiar solo lo que hace falta para que distinga abeja de velutina y no dispare nunca a la primera.

**AVISO MUY SERIO ARRIBA DEL TODO:** el láser aquí es **SIMULADO**. En el código verás `SIMULATED=True` por todas partes, clase 1M simulada y obturador simulado. Un láser de verdad clase 4 te deja ciego al instante y es ilegal en exterior sin certificación. Este repo **no es un arma** y no trae instrucciones para montar una. Si quieres pasar a hardware, habla con un laboratorio acreditado.

---

## Qué hay aquí y qué no

**Sí hay:**
- 6 bloques que funcionan en simulación, con 259 tests en verde
- Ojos: LiDAR + radar + cámara (simulados) que miden distancia, velocidad y color
- Cerebro: YOLOv8s (simulado + dos YOLOs reales entrenados con 380 fotos de GBIF) que distingue velutina de abeja/crabro
- Disparo simulado: galvo + lente + obturador fail-safe + control de energía
- Tiempo real simulado: scheduler 1ms, cola UART, interlock que corta en <0,1ms
- App simulada: panel, telemetría MQTT, OTA con vuelta atrás
- Plan de jaula virtual + checklist IEC simulado + trazabilidad MLOps
- Demo que puedes ver en tu terminal sin instalar nada raro

**No hay:**
- Láser real, ni instrucciones para montarlo
- Fotos con copyright metidas en el repo (las bajas tú si quieres)
- Promesas de que "funciona en campo" — en simulación va perfecto, en campo falta probarlo con bichos de verdad

---

## Cómo se ve

```bash
python sim/demo_apiario.py
```

Te sale el mapa del apiario en ASCII: `b` abejas en fila a la piquera `H`, `V` velutinas en hovering. En el frame 2 el sistema las pilla (mira las columnas `Pv/Pb`, `R1..R4`, `voto`) y las marca como `X`. Al final:

```
Velutinas neutralizadas: 3/3
Abejas tocadas: 0
Precisión: 100%
```

Para más chicha:

```bash
python sim/demo_apiario.py --sin-pausa --frames 10
python sim/eval_by_condition.py        # por lluvia / sol / oclusión / atardecer
python sim/monte_carlo.py --semillas 30  # media ± intervalo de confianza
python sim/sweep.py --eje distancia      # dónde empieza a fallar
python sim/stress.py                     # 100 bichos, sin LiDAR, shutter cerrado...
```

---

## Cómo funciona por dentro (en humano)

1. **Los ojos** miden dónde está el bicho y cuánto mide. Si no está entre 15 y 35mm, fuera.
2. **El radar** dice si está quieto en el aire (velutina acechando) o en línea recta (abeja entrando).
3. **La cámara** recorta 320x320 alrededor y mira color: tórax oscuro + banda naranja = sospechosa.
4. **La IA** (YOLO) da dos notas: `P(velutina)` y `P(abeja)`. Solo si `Pvel>=0.995` y `Pabe<=0.001` y pasan las 4 reglas, y el obturador está listo y no hay humano a <2m, entonces autoriza.
5. **El galvo** lleva el punto y el obturador simula clase 1M (<1,8µJ con obturador cerrado). Si el error de puntería >10mm, aborta.

Todo eso está partido en `jetson/` (fusión, IA, láser), `firmware/` (galvo/shutter), `sim/` (sensores, link, jitter) y `telemetry/`/`app/`.

---

## Instalar y probar (2 minutos)

```bash
git clone https://github.com/Manstor21/laser-antipest-sim.git
cd laser-antipest-sim
pip install -r requirements.txt  # si no hay, con numpy+scipy+pillow+opencv ya tiras
python -m pytest tests -q        # 259 passed
python sim/demo_apiario.py --sin-pausa
```

¿Quieres probar con fotos reales sin entrenar? Ya trae diagnóstico:

```bash
python sim/dataset/replay_real.py          # te dice qué descargar
python sim/dataset/replay_gbif.py --limit 150  # usa las 380 fotos GBIF si las bajaste a data/real/gbif_*
```

¿Quieres entrenar de verdad? Tienes 3 cuadernos listos en `notebooks/`:
- `train_yolo_colab.ipynb` — clasificador con 380 fotos GBIF, 0 MB en tu disco
- `train_yolo_colab_v2.ipynb` — el gordo con 7 especies y calibración FP=0
- `train_yolo_detect_kaggle_handsfree.ipynb` — detector con cajas Hornet3000, manos libres en Kaggle, apagas el PC y sigue (~1h, GPU gratis)

El detector local ya entrenado está en `jetson/ai/yolov8n_det.pt` (5,9 MB, 5 épocas, mAP50 0.84). Los de Colab/Kaggle van a `jetson/ai/yolov8s_velutina_v2.pt` etc.

---

## Versión barata para casa (solo avisa, 210€)

Si no quieres láser, la parte de visión sola sí la puedes montar sin riesgo:

- Raspberry Pi 5 8GB + SD + alimentador (~110€)
- Pi Camera Module 3 (30€, 50€ la global shutter)
- Benewake TF-Luna (45€, opcional)
- Caja IP65 + soporte (30€)

Te avisa al móvil cuando ve velutina y guarda recortes para re-entrenar. En `docs/DATASETS.md` y `docs/COLAB.md` tienes cómo recolectar y re-entrenar con tus fotos.

---

## Tests y papeles

Cada bloque tiene su `explore → proposal → spec → design → tasks → apply → verify → archive` guardado en memoria (Engram). Los `verify` son los que mandan:

- Bloque 1 fusión: 52 tests, 98%, precisión 1.0
- Bloque 2 IA: 76 tests, 15/15 casos
- Bloque 3 láser: 47 láser + 123 totales
- Bloque 4 tiempo real: 152 totales
- Bloque 5 app: 180 totales
- Bloque 6 jaula/IEC: 26 + 38 regresión

```bash
python -m pytest tests -q
```

---

## De dónde salen las fotos

- GBIF/iNaturalist (82k obs. velutina, filtradas a CC-BY, atribución en `data/real/gbif_ATTRIBUTION.csv` si las bajas)
- Kaggle Hornet3000+ (Observation.org, YOLO, 3088 train + 297 val)
- Roboflow Universe (Vespa velutina 3715, Honey Bee 909, etc.)

Respeta las licencias (CC-BY, pide atribución, mira uso comercial). No se suben fotos al repo.

---

## Licencia y créditos

MIT para el código. Las fotos y pesos que bajes mantienen su licencia original (mira `ATTRIBUTION.csv` y `docs/DATASETS.md`). Si usas esto en un colmenar, cuéntalo y comparte recortes anonimizados — entre todos se entrena mejor.

Hecho con calma, sin prisas, simulando primero. Si lo montas de verdad, hazlo con un técnico y con papeles. Y si ves una velutina, avisa al 112 apícola de tu zona antes que al láser.

