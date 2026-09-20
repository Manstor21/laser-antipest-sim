"""Backend detector YOLOv8n (cajas) — jetson/ai/det_backend.py

Usa jetson/ai/yolov8n_det.pt (copia stripped de runs/detect/train/weights/best.pt)
si existe. No toca detector.py ni yolo_backend.py (cls sigue por defecto).

API:
    load() -> dict con {"model": YOLO, "names": {...}, "path": str}
    detect(image) -> list[dict]  # cada dict: {"xyxy": [x1,y1,x2,y2], "conf": float, "cls": int, "name": str}

- image: numpy array HxWx3 (BGR o RGB) o path str; si es numpy asume BGR si no se especifica.
- Si no hay pesos -> FileNotFoundError claro (no se inventa).
- Si falta ultralytics -> ImportError claro.
- CPU por defecto (device="cpu"), conf 0.25, iou 0.45, imgsz 320.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Union

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "jetson" / "ai" / "yolov8n_det.pt"
# fallback a best.pt si el stripped no estuviera (compat)
FALLBACK_BEST = REPO_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"

_cache = None


def _get_model_path() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH
    # no fallback automático a best.pt si supera 20MB? Pero si el usuario entrenó y no copió, avisamos claro
    return MODEL_PATH


def load() -> dict:
    """Carga YOLO detector una vez y lo cachea. Sin deps o sin .pt -> error claro."""
    global _cache
    if _cache is not None:
        return _cache
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Falta ultralytics para el detector YOLO. "
            "Instala con: pip install ultralytics (gratis)."
        ) from exc
    p = _get_model_path()
    if not p.exists():
        raise FileNotFoundError(
            f"No existe el peso detector ({p}). "
            f"Entrena primero: python sim/dataset/train_det.py o copia runs/detect/train/weights/best_stripped.pt a {MODEL_PATH}."
        )
    # check size warning
    size_mb = p.stat().st_size / (1024 * 1024)
    if size_mb > 20:
        raise ValueError(f"Peso detector demasiado grande ({size_mb:.1f} MB >20MB), revisa {p}")
    try:
        model = YOLO(str(p))
    except Exception as exc:
        raise RuntimeError(f"El peso detector no carga ({exc}).") from exc
    names: dict = dict(getattr(model, "names", {}))
    _cache = {"model": model, "names": names, "path": str(p)}
    return _cache


def detect(
    image: Union[str, Path, np.ndarray],
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 320,
    device: str = "cpu",
) -> List[Dict]:
    """Detecta cajas. Retorna lista (puede vacía) sin error.

    Cada elemento: {"xyxy": [x1,y1,x2,y2] (float, pixeles en imagen original),
                   "xywhn": [cx,cy,w,h] (norm), "conf": float, "cls": int, "name": str}
    """
    m = load()
    model = m["model"]
    # ultralytics acepta numpy, path, PIL
    results = model.predict(image, conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False)
    if not results:
        return []
    r = results[0]
    out: List[Dict] = []
    if r.boxes is None or len(r.boxes) == 0:
        return out
    # r.boxes.xyxyn for normalized, r.boxes.xyxy for pixel, r.boxes.xywhn
    try:
        xyxy = r.boxes.xyxy.cpu().numpy()  # pixel coords in input size? Actually original image size
        xywhn = r.boxes.xywhn.cpu().numpy() if hasattr(r.boxes, "xywhn") else None
        clss = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        names = m["names"]
        for i in range(len(clss)):
            box = xyxy[i].tolist() if xyxy is not None else [0, 0, 0, 0]
            whn = xywhn[i].tolist() if xywhn is not None else [0, 0, 0, 0]
            cls = int(clss[i])
            out.append(
                {
                    "xyxy": [float(x) for x in box],
                    "xywhn": [float(x) for x in whn],
                    "conf": float(confs[i]),
                    "cls": cls,
                    "name": str(names.get(cls, cls)),
                }
            )
    except Exception as exc:
        raise RuntimeError(f"Error al decodificar cajas ({exc})") from exc
    return out


def reset_cache() -> None:
    global _cache
    _cache = None
