"""Backend de inferencia del modelo REAL (solo numpy, sin sklearn).

`load()` lee jetson/ai/real_model.npz una sola vez (cache).
`predict_proba(crop 320x320x3)` devuelve (Pv, Pb=1-Pv) con sigmoide manual
sobre features IDENTICAS a las de sim/dataset/train_real.py.

Sin el npz -> error claro (hay que entrenar primero con train_real.py).
NO toca el stub detector (jetson/ai/detector.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "jetson" / "ai" / "real_model.npz"

N_H, N_S, N_V = 16, 8, 8
FEATURE_NAMES = (
    [f"H{i:02d}" for i in range(N_H)]
    + [f"S{i}" for i in range(N_S)]
    + [f"V{i}" for i in range(N_V)]
    + ["dark_ratio", "bright_ratio", "edge_density"]
)

_cache: dict | None = None


def extract_features(crop_320: np.ndarray) -> np.ndarray:
    """35 features IDENTICAS a train_real.extract_features."""
    import cv2

    img = np.ascontiguousarray(crop_320, dtype=np.uint8)
    if img.shape != (320, 320, 3):
        raise ValueError(f"crop 320x320x3 requerido, llego {img.shape}")
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    n_pix = float(hsv.shape[0] * hsv.shape[1])
    h = cv2.calcHist([hsv], [0], None, [N_H], [0, 180]).ravel() / n_pix
    s = cv2.calcHist([hsv], [1], None, [N_S], [0, 256]).ravel() / n_pix
    v = cv2.calcHist([hsv], [2], None, [N_V], [0, 256]).ravel() / n_pix
    vv = hsv[:, :, 2]
    dark_ratio = float((vv < 60).mean())
    bright_ratio = float((vv > 200).mean())
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    edge_density = float(np.mean(mag)) / 255.0
    return np.concatenate([h, s, v, [dark_ratio, bright_ratio, edge_density]])


def _sigmoid(z: float) -> float:
    z = float(np.clip(z, -500.0, 500.0))
    return 1.0 / (1.0 + float(np.exp(-z)))


def load() -> dict:
    """Carga el npz una vez y lo cachea. Sin npz -> FileNotFoundError claro."""
    global _cache
    if _cache is not None:
        return _cache
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No existe el modelo entrenado ({MODEL_PATH}). "
            "Entrena primero con: python sim/dataset/train_real.py"
        )
    z = np.load(str(MODEL_PATH), allow_pickle=False)
    coef = np.asarray(z["coef"], dtype=np.float64).ravel()
    intercept = float(np.asarray(z["intercept"]).ravel()[0])
    mean = np.asarray(z["mean"], dtype=np.float64).ravel()
    scale = np.asarray(z["scale"], dtype=np.float64).ravel()
    names = [str(n) for n in np.asarray(z["feature_names"]).ravel().tolist()]
    if names != FEATURE_NAMES:
        raise ValueError(
            f"Orden de features del npz no coincide con el esperado: {names} vs {FEATURE_NAMES}"
        )
    if coef.shape != (35,) or mean.shape != (35,) or scale.shape != (35,):
        raise ValueError(f"Formas inesperadas en {MODEL_PATH}: coef={coef.shape}")
    thr = float(np.asarray(z["threshold_fp0"]).ravel()[0])
    _cache = {"coef": coef, "intercept": intercept, "mean": mean,
              "scale": scale, "threshold_fp0": thr, "path": str(MODEL_PATH)}
    return _cache


def predict_proba(crop_320: np.ndarray) -> tuple[float, float]:
    """(Pv, Pb=1-Pv) desde un crop RGB 320x320x3 uint8."""
    m = load()
    feats = extract_features(crop_320)
    znorm = (feats - m["mean"]) / m["scale"]
    logit = float(m["coef"] @ znorm + m["intercept"])
    pv = _sigmoid(logit)
    return pv, 1.0 - pv


def reset_cache() -> None:  # util para tests
    global _cache
    _cache = None
