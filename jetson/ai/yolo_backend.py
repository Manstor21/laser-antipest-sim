"""Backend de inferencia YOLOv8n-cls (OPCIONAL, diagnostico).

`load()` carga jetson/ai/yolov8n_velutina.pt una sola vez (cache).
`predict_proba(crop 320x320x3)` devuelve (Pv, Pb=1-Pv):
  Pv = prob. softmax de la clase 'velutina' (3 clases: abeja/crabro/velutina),
  Pb = todo lo no-velutina (abeja+crabro), filosofia duda->abeja.

- Resize interno 320->224 (cv2 si esta, si no Pillow).
- Sin ultralytics/torch o sin el .pt -> error claro (no se inventa nada).
- NO toca detector.py ni thresholds.yaml: el stub sigue por defecto.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "jetson" / "ai" / "yolov8n_velutina.pt"

_cache: dict | None = None


def _to_224(crop_320: np.ndarray) -> np.ndarray:
    img = np.ascontiguousarray(crop_320, dtype=np.uint8)
    if img.shape != (320, 320, 3):
        raise ValueError(f"crop 320x320x3 requerido, llego {img.shape}")
    try:
        import cv2

        return np.ascontiguousarray(cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR))
    except ImportError:
        from PIL import Image

        return np.asarray(Image.fromarray(img).resize((224, 224), Image.BILINEAR))


def load() -> dict:
    """Carga el YOLO una vez y lo cachea. Sin deps o sin .pt -> error claro."""
    global _cache
    if _cache is not None:
        return _cache
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Falta ultralytics para el backend YOLO. "
            "Instala con: pip install ultralytics (gratis)."
        ) from exc
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No existe el peso YOLO ({MODEL_PATH}). "
            "Copialo desde DESCARGAS/best.pt (solo si <10MB)."
        )
    try:
        model = YOLO(str(MODEL_PATH))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"El peso YOLO no carga ({exc}).") from exc
    names: dict = dict(getattr(model, "names", {}))
    idx_of = {v: k for k, v in names.items()}
    if "velutina" not in idx_of:
        raise ValueError(f"El modelo no tiene clase 'velutina' (names={names}).")
    _cache = {"model": model, "idx_velutina": idx_of["velutina"], "path": str(MODEL_PATH)}
    return _cache


def predict_proba(crop_320: np.ndarray) -> tuple[float, float]:
    """(Pv, Pb=1-Pv) desde un crop RGB 320x320x3 uint8. Determinista en CPU."""
    m = load()
    img224 = _to_224(crop_320)
    res = m["model"].predict(img224, verbose=False, device="cpu")[0]
    probs = np.asarray(res.probs.data.cpu()).ravel().astype(np.float64)
    pv = float(probs[int(m["idx_velutina"])])
    pv = min(1.0, max(0.0, pv))
    return pv, 1.0 - pv


def reset_cache() -> None:  # util para tests
    global _cache
    _cache = None
