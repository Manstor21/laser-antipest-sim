"""Entrena un modelo ENTRENADO DE VERDAD con las 380 fotos GBIF reales.

Uso:
    python sim/dataset/train_real.py [--limit N] [--timebox-min 20]

- Carga data/real/gbif_velutina | gbif_abeja | gbif_crabro (crop central
  cuadrado -> 320 como en sim/dataset/replay_gbif.py; reutiliza por
  importacion y si no duplica lo minimo).
- Features por imagen (~35 numeros, numpy/cv2, rapidas):
  hist H(16)+S(8)+V(8) normalizado + dark_ratio (V<60) + bright_ratio (V>200)
  + densidad de bordes (Sobel, media de magnitud) = 35.
- Etiquetas: velutina=1, abeja/crabro=0. Split 80/20 estratificado (semilla 42).
- LogisticRegression(max_iter=2000).
- Evalua en val: recall, FP abeja, FP crabro, precision + barrido de umbral
  0.5->0.999 para encontrar el umbral con FP_total=0 y su recall
  (filosofia duda->abeja).
- Exporta pesos a jetson/ai/real_model.npz.

NO toca el stub detector (jetson/ai/detector.py) ni ningun fichero existente.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_DIRS = {
    "gbif_velutina": (REPO_ROOT / "data" / "real" / "gbif_velutina", 1),
    "gbif_abeja": (REPO_ROOT / "data" / "real" / "gbif_abeja", 0),
    "gbif_crabro": (REPO_ROOT / "data" / "real" / "gbif_crabro", 0),
}
OUT_NPZ = REPO_ROOT / "jetson" / "ai" / "real_model.npz"

N_H, N_S, N_V = 16, 8, 8
FEATURE_NAMES = (
    [f"H{i:02d}" for i in range(N_H)]
    + [f"S{i}" for i in range(N_S)]
    + [f"V{i}" for i in range(N_V)]
    + ["dark_ratio", "bright_ratio", "edge_density"]
)
assert len(FEATURE_NAMES) == 35, len(FEATURE_NAMES)


# --- Carga de imagenes: reutiliza replay_gbif si es importable ---------------
try:
    from sim.dataset.replay_gbif import (  # type: ignore
        glob_images as _glob_images,
        load_image_rgb as _load_image_rgb,
        make_center_crop_320 as _make_center_crop_320,
    )

    _REUSED = "sim.dataset.replay_gbif (importado)"
except Exception:  # pragma: no cover - fallback minimo duplicado
    _glob_images = None
    _load_image_rgb = None
    _make_center_crop_320 = None
    _REUSED = None


def _fallback_load_image_rgb(img_path: Path) -> np.ndarray:
    try:
        import cv2

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is not None:
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except ImportError:
        pass
    from PIL import Image

    with Image.open(img_path) as im:
        return np.asarray(im.convert("RGB"))


def _fallback_center_crop_320(img_rgb) -> np.ndarray:
    img = np.asarray(img_rgb)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    img = np.ascontiguousarray(img, dtype=np.uint8)
    H, W = img.shape[:2]
    if H >= 320 and W >= 320:
        y0, x0 = (H - 320) // 2, (W - 320) // 2
        return np.ascontiguousarray(img[y0:y0 + 320, x0:x0 + 320])
    scale = 320.0 / max(H, W) if max(H, W) < 320 else 320.0 / min(H, W)
    new_w = max(1, int(round(W * scale)))
    new_h = max(1, int(round(H * scale)))
    try:
        import cv2

        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        from PIL import Image

        img = np.asarray(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))
    H, W = img.shape[:2]
    canvas = np.zeros((320, 320, 3), dtype=np.uint8)
    y0, x0 = (H - 320) // 2, (W - 320) // 2
    sy0, sy1 = max(0, y0), min(H, y0 + 320)
    sx0, sx1 = max(0, x0), min(W, x0 + 320)
    dy0 = sy0 - y0 if y0 < 0 else 0
    dx0 = sx0 - x0 if x0 < 0 else 0
    if sy1 > sy0 and sx1 > sx0:
        canvas[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = img[sy0:sy1, sx0:sx1]
    return np.ascontiguousarray(canvas)


def _fallback_glob_images(folder: Path):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    found: list[Path] = []
    if folder.is_dir():
        for ext in exts:
            found.extend(sorted(folder.glob(f"*{ext}")))
            found.extend(sorted(folder.glob(f"*{ext.upper()}")))
    seen, out = set(), []
    for p in found:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return sorted(out)


glob_images = _glob_images or _fallback_glob_images
load_image_rgb = _load_image_rgb or _fallback_load_image_rgb
make_center_crop_320 = _make_center_crop_320 or _fallback_center_crop_320
LOAD_SOURCE = _REUSED or "duplicado minimo local (replay_gbif no importable)"


# --- Features (DEBEN ser identicas en jetson/ai/real_backend.py) -------------
def extract_features(crop_320: np.ndarray) -> np.ndarray:
    """35 features desde un crop 320x320x3 RGB uint8."""
    import cv2

    img = np.ascontiguousarray(crop_320, dtype=np.uint8)
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


def load_dataset(limit: int | None):
    X_list, y_list, groups = [], [], []
    counts = {}
    for dirname, (folder, label) in DATA_DIRS.items():
        paths = glob_images(Path(folder))
        if limit:
            paths = paths[:limit]
        counts[dirname] = len(paths)
        for p in paths:
            try:
                img = load_image_rgb(p)
                crop = make_center_crop_320(img)
            except Exception as exc:  # noqa: BLE001
                print(f"Aviso: no se pudo leer {p} ({exc}), se omite.", file=sys.stderr)
                continue
            X_list.append(extract_features(crop))
            y_list.append(label)
            groups.append(dirname)
    X = np.asarray(X_list, dtype=np.float64)
    y = np.asarray(y_list, dtype=np.int64)
    groups = np.asarray(groups)
    return X, y, groups, counts


def metrics_at_threshold(y_true, proba, thr, groups):
    pred = (proba >= thr).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fp_abeja = int(((pred == 1) & (groups == "gbif_abeja")).sum())
    fp_crabro = int(((pred == 1) & (groups == "gbif_crabro")).sum())
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    return {
        "thr": thr, "tp": tp, "fn": fn, "fp_total": fp,
        "fp_abeja": fp_abeja, "fp_crabro": fp_crabro,
        "recall": recall, "precision": precision,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Entrena modelo real GBIF (regresion logistica 35 features).")
    ap.add_argument("--limit", type=int, default=0, help="Max fotos por clase (0 = todas)")
    ap.add_argument("--timebox-min", type=float, default=20.0, help="Timebox minutos")
    args = ap.parse_args(argv)
    limit = args.limit or None

    t0 = time.time()
    try:
        import sklearn  # noqa: F401
    except ImportError:
        print("ERROR: falta scikit-learn. Instala con: pip install scikit-learn (gratis).",
              file=sys.stderr)
        print("Aborto: no se puede entrenar sin sklearn; no se inventa ningun resultado.",
              file=sys.stderr)
        return 3
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    print(f"Carga de imagenes via: {LOAD_SOURCE}")
    X, y, groups, counts = load_dataset(limit)
    n_total = len(y)
    if n_total == 0:
        print("ERROR: no se cargaron fotos de data/real/gbif_*.", file=sys.stderr)
        return 2
    print(f"Fotos por carpeta: {counts} (total={n_total})")

    Xtr, Xva, ytr, yva, gtr, gva = train_test_split(
        X, y, groups, test_size=0.2, random_state=42, stratify=y
    )
    mean = Xtr.mean(axis=0)
    scale = Xtr.std(axis=0)
    scale[scale < 1e-6] = 1.0
    Ztr, Zva = (Xtr - mean) / scale, (Xva - mean) / scale

    clf = LogisticRegression(max_iter=2000)
    clf.fit(Ztr, ytr)
    pva = clf.predict_proba(Zva)[:, 1]

    m05 = metrics_at_threshold(yva, pva, 0.5, gva)
    grid = np.concatenate([np.arange(0.5, 0.9, 0.05), np.arange(0.9, 0.99, 0.01),
                           np.arange(0.99, 0.9991, 0.001)])
    rows = [metrics_at_threshold(yva, pva, float(t), gva) for t in grid]
    fp0 = [r for r in rows if r["fp_total"] == 0]
    best_fp0 = min(fp0, key=lambda r: r["thr"]) if fp0 else None

    elapsed = (time.time() - t0) / 60.0
    if elapsed > args.timebox_min:
        print(f"AVISO timebox: {elapsed:.1f} min > {args.timebox_min:.0f} min. "
              f"Repite con --limit por clase (p. ej. --limit 60).", file=sys.stderr)

    n_abeja_va = int((gva == "gbif_abeja").sum())
    n_crabro_va = int((gva == "gbif_crabro").sum())
    n_vel_va = int((yva == 1).sum())
    print("")
    print("Informe modelo real GBIF (lenguaje llano)")
    print("=" * 60)
    print("Modelo: regresion logistica con 35 features (hist H16+S8+V8 + dark/bright + bordes).")
    print(f"Train={len(ytr)} fotos, val={len(yva)} fotos (80/20 estratificado, semilla 42).")
    print(f"Val: {n_vel_va} velutina, {n_abeja_va} abeja, {n_crabro_va} crabro.")
    print(f"A umbral 0.5: recall={m05['recall']:.3f} ({m05['tp']}/{m05['tp'] + m05['fn']}), "
          f"FP abeja={m05['fp_abeja']}/{n_abeja_va}, FP crabro={m05['fp_crabro']}/{n_crabro_va}, "
          f"precision={m05['precision']:.3f}.")
    if best_fp0:
        print(f"Umbral con FP_total=0: thr={best_fp0['thr']:.3f} -> recall={best_fp0['recall']:.3f} "
              f"({best_fp0['tp']}/{best_fp0['tp'] + best_fp0['fn']}). Filosofia duda->abeja: "
              f"a ese umbral ninguna abeja/crabro de val se confunde con velutina.")
    else:
        print("AVISO honesto: ningun umbral del barrido 0.5->0.999 deja FP_total=0 en val; "
              "se guarda thr=1.0 (nadie promociona).")
    print(f"Tiempo total: {elapsed:.1f} min. Como leerlo: recall alto = pillamos casi todas "
          f"las velutinas; FP = bichos que confundimos; a umbral FP0 no hay falsas alarmas en val "
          f"pero se escapan mas velutinas.")

    thr_fp0 = float(best_fp0["thr"]) if best_fp0 else 1.0
    rec_fp0 = float(best_fp0["recall"]) if best_fp0 else 0.0
    np.savez(
        OUT_NPZ,
        coef=clf.coef_.ravel().astype(np.float64),
        intercept=np.asarray([float(clf.intercept_.ravel()[0])]),
        mean=mean.astype(np.float64),
        scale=scale.astype(np.float64),
        feature_names=np.asarray(FEATURE_NAMES),
        threshold_fp0=np.asarray([thr_fp0]),
        recall_05=np.asarray([float(m05["recall"])]),
        precision_05=np.asarray([float(m05["precision"])]),
        fp_abeja_05=np.asarray([m05["fp_abeja"]]),
        fp_crabro_05=np.asarray([m05["fp_crabro"]]),
        recall_fp0=np.asarray([rec_fp0]),
        n_train=np.asarray([len(ytr)]),
        n_val=np.asarray([len(yva)]),
    )
    print(f"Pesos guardados en: {OUT_NPZ}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
