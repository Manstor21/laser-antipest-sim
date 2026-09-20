# Laser Anti Plagas

Sistema de detección y clasificación de plagas voladoras basado en fusión multisensor y visión por computador. Simulación completa, sin hardware dedicado, orientado a la protección de colmenares frente a *Vespa velutina* y adaptable a otras especies.

> **Nota de seguridad:** Este repositorio contiene **únicamente simulación**. Todos los módulos relacionados con emisión láser operan con `SIMULATED=True`, sin generación de radiación real, sin valor certificante y sin capacidad de disparo físico. Cualquier implementación con hardware láser requiere evaluación por laboratorio acreditado conforme a IEC 60825-1 y normativa aplicable.

---

## Índice

1. [Resumen](#resumen)
2. [Arquitectura](#arquitectura)
3. [Módulos](#módulos)
4. [Capacidades actuales](#capacidades-actuales)
5. [Instalación](#instalación)
6. [Uso](#uso)
7. [Entrenamiento con datos reales](#entrenamiento-con-datos-reales)
8. [Variante de bajo coste para colmenar](#variante-de-bajo-coste-para-colmenar)
9. [Verificación y calidad](#verificación-y-calidad)
10. [Estructura del repositorio](#estructura-del-repositorio)
11. [Datasets y licencias](#datasets-y-licencias)
12. [Roadmap](#roadmap)
13. [Licencia](#licencia)

---

## Resumen

Laser Anti Plagas integra, en un pipeline determinista, la localización espacial (LiDAR + radar), la caracterización morfológica y cromática (cámara) y la clasificación mediante modelos de aprendizaje profundo (YOLO), con lógica de decisión orientada a **selectividad**: el sistema prioriza no afectar a *Apis mellifera* frente a maximizar la tasa de detección del objetivo.

El objetivo inicial es *Vespa velutina* en entorno de apiario. La parametrización de especies, umbrales y geometría es externa (ficheros YAML), por lo que el mismo sistema es aplicable a otras plagas mediante ajuste de configuración y reentrenamiento, sin modificar el código.

El estado actual es **sim-only**: todos los sensores, actuadores y enlaces se ejecutan mediante modelos sintéticos deterministas. Esto permite validar la cadena completa, reproducir escenarios y medir métricas sin coste de hardware.

---

## Arquitectura

```
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│ Sensores          │     │ Fusión y          │     │ Clasificación     │     │ Decisión y        │
│  LiDAR 3D         │────▶│  seguimiento      │────▶│  YOLOv8           │────▶│  actuación sim.   │
│  Radar FMCW       │     │  EKF 9-DOF        │     │  voto 3/3         │     │  control + inter. │
│  Cámara 320px     │     │  predicción 50ms  │     │  calibración      │     │  galvo simulado   │
└───────────────────┘     └───────────────────┘     └───────────────────┘     └───────────────────┘
                                  │                       │                       │
                                  └───────────┬───────────┘                       │
                                              ▼                                   ▼
                                   ┌───────────────────┐               ┌───────────────────┐
                                   │ Telemetría        │               │ App / panel       │
                                   │  MQTT sim         │               │  + logs JSONL     │
                                   │  OTA con rollback │               │                   │
                                   └───────────────────┘               └───────────────────┘
```

Principios de diseño:

- **Configuración externa.** Umbrales, clases y parámetros de sensores en `jetson/fusion/config.yaml` y `jetson/ai/thresholds.yaml`.
- **Determinismo.** Semillas fijas y tiempo simulado explícito para reproducibilidad de ensayos.
- **Seguridad por diseño.** Autorización de disparo condicionada a conjunción de criterios (tamaño, cinemática, color, clasificación y estado de interlocks). Ante duda, el sistema clasifica como especie protegida y no autoriza.
- **Trazabilidad.** Registro de versiones de modelo, manifiestos de datos y logs deterministas por ejecución.

---

## Módulos

### 1. Fusión y seguimiento

- EKF de 9 estados con predicción a 50 ms para compensar latencia.
- Gestor de ROI de 320×320 y mecanismo de retorno a campo completo tras pérdidas consecutivas.
- Reglas R1–R4 (tamaño, velocidad, color, morfología) como barreras previas a la clasificación.

### 2. Inteligencia artificial

- Detector de referencia YOLOv8s-seg sobre ROI, con votación temporal 2 de 3 y veto por clase protegida.
- Calibración por escalado de temperatura y doble umbral `P(objetivo) ≥ 0,995 ∧ P(protegida) ≤ 0,001`.
- Soporte para modelos reales entrenados con datos GBIF y Hornet3000+, con inferencia desacoplada y evaluación dedicada.

### 3. Actuación y óptica simuladas

- Controlador de disparo con máquina de estados, medición de ciclo de trabajo y límite de ráfaga.
- Modelo de galvo de segundo orden y óptica f-theta con control de fluencia.
- Obturador normalmente cerrado con watchdog y verificación de atenuación.

### 4. Tiempo real simulado

- Núcleo discreto con tick de 1 ms, prioridades definidas y medición de jitter.
- Simulación de enlace serie (latencia y descarte por cola llena) e interlock hardware independiente en tiempo simulado.

### 5. Telemetría y aplicación

- Bus MQTT en memoria, panel de estado, historial JSONL y estimación de balance de enlace LoRa/WiFi.
- Actualización OTA basada en metadatos con rollback retenido.

### 6. Validación

- Jaula virtual, ensayos adversos (lluvia, deslumbramiento, oclusión, penumbra), matriz de selectividad y lista de verificación de seguridad simulada.

---

## Capacidades actuales

- Detección y seguimiento en escenarios sintéticos con métricas por condición.
- Clasificación con modelos de referencia y modelos entrenados con datos reales (GBIF/Hornet3000).
- Evaluación de selectividad y balance de enlace mediante ensayos reproducibles.
- Conjunto de verificación de 259 pruebas automatizadas.

Limitaciones conocidas:

- Modelos entrenados sobre recortes centrados; la transferencia a escenas completas requiere recalibración.
- Sensores y actuadores simulados; no sustituye ensayos de campo ni certificación.

---

## Instalación

Requisitos: Python 3.10+, pip.

```bash
git clone https://github.com/Manstor21/laser-antipest-sim.git
cd laser-antipest-sim
pip install -r requirements.txt
python -m pytest tests -q
```

Dependencias principales: `numpy`, `scipy`, `pillow`, `opencv-python`, `pyyaml`, `scikit-learn`, `ultralytics` (opcional, para modelos reales).

---

## Uso

### Demostración de apiario

```bash
python sim/demo_apiario.py
python sim/demo_apiario.py --sin-pausa --frames 10
```

### Evaluación

```bash
python sim/eval_by_condition.py
python sim/monte_carlo.py --semillas 30
python sim/sweep.py --eje distancia
python sim/stress.py
python sim/dataset/replay_gbif.py --limit 150
python sim/dataset/replay_boxes.py --split val
python sim/dataset/replay_boxes.py --split val --sweep
```

### Entrenamiento

Cuadernos en `notebooks/` para ejecución en Colab o Kaggle sin coste local:

- `train_yolo_colab.ipynb` y `train_yolo_colab_v2.ipynb` — clasificación con datos GBIF.
- `train_yolo_detect_kaggle_handsfree.ipynb` — detección con Hornet3000, ejecución desatendida en Kaggle (GPU gratuita).

La documentación de cada cuaderno detalla pasos y salidas esperadas en `docs/COLAB.md` y `docs/KAGGLE.md`.

---

## Entrenamiento con datos reales

El repositorio no incluye imágenes con copyright. El flujo previsto es:

1. Descarga de conjuntos abiertos (GBIF, Hornet3000+, Roboflow) según `docs/DATASETS.md`.
2. Generación de recortes y manifiestos mediante `sim/dataset/`.
3. Entrenamiento externo (Colab/Kaggle) y evaluación con `sim/dataset/eval_yolo.py` y `sim/dataset/replay_boxes.py`.

Los modelos resultantes se almacenan fuera del control de versiones y se referencian por hash en los manifiestos.

---

## Variante de bajo coste para colmenar

Configuración de aviso sin actuación, basada en Raspberry Pi 5, cámara y caja estanca. Coste estimado 210–250 €. El sistema registra recortes y genera avisos; el entrenamiento se realiza externamente y los pesos se despliegan por OTA simulada. Detalle en `docs/` y en la sección de arquitectura.

---

## Verificación y calidad

- 259 pruebas automatizadas. Ejecución: `python -m pytest tests -q`.
- Verificación por bloques con informes de cumplimiento de requisitos y escenarios.
- Ensayos deterministas por semilla y por condición (lluvia, oclusión, deslumbramiento, penumbra).

---

## Estructura del repositorio

```
app/                 Panel y gestión de configuración simulada
docs/                Guías de datos, entrenamiento y matrices de verificación
firmware/            Modelos de MCU, galvo y obturador (simulados)
jetson/              Fusión, IA y control de disparo
notebooks/           Cuadernos de entrenamiento Colab/Kaggle
sim/                 Sensores, escenarios y herramientas de evaluación
telemetry/           Bus MQTT, balance de enlace y OTA
tests/               Pruebas unitarias, de integración y de sistema
```

---

## Datasets y licencias

- GBIF / iNaturalist — registros con licencias Creative Commons (atribución requerida, ver `ATTRIBUTION.csv` tras descarga).
- Hornet3000+ (Observation.org) — anotaciones YOLO.
- Conjuntos adicionales vía Roboflow Universe.

Cada conjunto mantiene su licencia original. El código del repositorio se distribuye bajo licencia MIT.

---

## Roadmap

- Recalibración de umbrales sobre recortes por caja y ampliación del conjunto de validación.
- Entrenamiento de detector con mayor número de épocas y validación en escena completa.
- Adaptación del modo recolecta para despliegue en Raspberry Pi y ciclo de mejora continua con datos propios.

---

## Licencia

MIT. Ver `LICENSE` para detalle. El uso con hardware de emisión requiere cumplimiento de la normativa aplicable y no está cubierto por este repositorio de simulación.
