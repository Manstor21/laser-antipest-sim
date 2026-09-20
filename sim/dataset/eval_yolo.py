"""Evalua el YOLOv8n-cls de Colab con las 380 fotos GBIF locales.

Uso:
    python sim/dataset/eval_yolo.py [--weights DESCARGAS/best.pt]
        [--limit N] [--dirs gbif_velutina,gbif_abeja,gbif_crabro]
        [--umbral 0.99]

- Crop CENTRAL cuadrado -> 224 (cv2 si esta, si no Pillow), igual filosofia
  que train_real.py pero a 224 (tamano del clasificador YOLO).
- `model.predict(verbose=False)` por lote; top1 por foto.
- Informa: top1 global, recall velutina, FP abeja->velutina,
  FP crabro->velutina, top1 por clase + matriz de confusion ASCII.
- Con --umbral U (p. ej. 0.99, el de operating_point.json del cuaderno v2):
  aplica Pv>=U como promote y reporta recall/FP a ese umbral.
  Sin --umbral el informe es identico al de siempre.
- Compara con DESCARGAS/resultados.txt de Colab (deben parecerse) y con el
  logistico local (recall 0.70, FP0@0.990 recall 0.133, docs/REAL_MODEL.md).

NO entrena, NO modifica pesos, NO toca el stub detector.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_WEIGHTS = REPO_ROOT / "DESCARGAS" / "best.pt"
COLAB_TXT = REPO_ROOT / "DESCARGAS" / "resultados.txt"
DATA_ROOT = REPO_ROOT / "data" / "real"

TRUE_OF_DIR = {
    "gbif_velutina": "velutina",
    "gbif_abeja": "abeja",
    "gbif_crabro": "crabro",
}
ALL_DIRS = ["gbif_velutina", "gbif_abeja", "gbif_crabro"]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def glob_images(folder: Path) -> list[Path]:
    found: list[Path] = []
    if folder.is_dir():
        for ext in IMG_EXTS:
            found.extend(sorted(folder.glob(f"*{ext}")))
            found.extend(sorted(folder.glob(f"*{ext.upper()}")))
    seen, out = set(), []
    for p in found:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return sorted(out)


def load_rgb(path: Path) -> np.ndarray:
    try:
        import cv2

        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is not None:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except ImportError:
        pass
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def center_crop_224(img_rgb: np.ndarray) -> np.ndarray:
    img = np.asarray(img_rgb)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    img = np.ascontiguousarray(img, dtype=np.uint8)
    h, w = img.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    sq = img[y0:y0 + side, x0:x0 + side]
    try:
        import cv2

        return np.ascontiguousarray(cv2.resize(sq, (224, 224), interpolation=cv2.INTER_LINEAR))
    except ImportError:
        from PIL import Image

        return np.asarray(Image.fromarray(sq).resize((224, 224), Image.BILINEAR))


def parse_colab_txt(path: Path) -> dict:
    vals: dict = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                k, v = k.strip(), v.strip()
                try:
                    vals[k] = float(v)
                except ValueError:
                    vals[k] = v
    except OSError:
        pass
    return vals


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evalua YOLOv8n-cls con fotos GBIF locales.")
    ap.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--limit", type=int, default=0, help="Max fotos por clase (0 = todas)")
    ap.add_argument("--dirs", type=str, default=",".join(ALL_DIRS),
                    help="Subcarpetas de data/real separadas por coma")
    ap.add_argument("--umbral", type=float, default=None,
                    help="Si se da (p. ej. 0.99), aplica Pv>=umbral como promote "
                    "y reporta recall/FP a ese umbral. Sin el, todo igual que antes.")
    args = ap.parse_args(argv)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: falta ultralytics. Instala con: pip install ultralytics (gratis).",
              file=sys.stderr)
        return 3

    w = Path(args.weights)
    if not w.is_file():
        print(f"ERROR: no existe el peso {w}.", file=sys.stderr)
        return 2
    dirs = [d.strip() for d in args.dirs.split(",") if d.strip()]
    dirs = [d for d in dirs if d in TRUE_OF_DIR]
    if not dirs:
        print("ERROR: --dirs no contiene ninguna carpeta conocida.", file=sys.stderr)
        return 2

    try:
        model = YOLO(str(w))
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: best.pt no carga ({exc}). Aborto sin inventar resultados.",
              file=sys.stderr)
        return 2
    names: dict = dict(getattr(model, "names", {}))
    idx_of = {v: k for k, v in names.items()}
    for need in ("abeja", "crabro", "velutina"):
        if need not in idx_of:
            print(f"ERROR: el modelo no tiene la clase '{need}' (names={names}).",
                  file=sys.stderr)
            return 2

    # Carga fotos
    crops: list[np.ndarray] = []
    true_names: list[str] = []
    counts: dict[str, int] = {}
    for d in dirs:
        paths = glob_images(DATA_ROOT / d)
        if args.limit:
            paths = paths[:args.limit]
        counts[d] = len(paths)
        for p in paths:
            try:
                crops.append(center_crop_224(load_rgb(p)))
            except Exception as exc:  # noqa: BLE001
                print(f"Aviso: no se pudo leer {p} ({exc}), se omite.", file=sys.stderr)
                continue
            true_names.append(TRUE_OF_DIR[d])
    n = len(crops)
    if n == 0:
        print("ERROR: no se cargaron fotos de data/real/gbif_*.", file=sys.stderr)
        return 2
    print(f"Fotos por carpeta: {counts} (total={n})")

    # Prediccion por lote (CPU); fallback a 1-a-1 si falla
    try:
        results = model.predict(crops, verbose=False, device="cpu")
        prob_list = [np.asarray(r.probs.data.cpu()).ravel() for r in results]
    except Exception:
        prob_list = []
        for c in crops:
            r = model.predict(c, verbose=False, device="cpu")[0]
            prob_list.append(np.asarray(r.probs.data.cpu()).ravel())
    order = [names[i] for i in sorted(names)]
    pred_names = [order[int(np.argmax(p))] for p in prob_list]

    classes = ["abeja", "crabro", "velutina"]
    conf: dict[tuple[str, str], int] = {(t, p): 0 for t in classes for p in classes}
    extras: dict[tuple[str, str], int] = {}
    for t, p in zip(true_names, pred_names):
        if (t, p) in conf:
            conf[(t, p)] += 1
        else:
            # Modelo v2 con mas clases (vespula, bombus...): se cuenta aparte
            # sin tocar el informe clasico de 3 clases.
            extras[(t, p)] = extras.get((t, p), 0) + 1
    n_true = {c: sum(1 for t in true_names if t == c) for c in classes}
    top1 = {c: (conf[(c, c)] / n_true[c] if n_true[c] else 0.0) for c in classes}
    top1_global = sum(1 for t, p in zip(true_names, pred_names) if t == p) / n
    recall_vel = top1["velutina"]
    fp_abeja = conf[("abeja", "velutina")]
    fp_crabro = conf[("crabro", "velutina")]

    print("")
    print("Informe YOLO local (lenguaje llano)")
    print("=" * 60)
    print(f"Peso: {w} | fotos evaluadas: {n} "
          f"(velutina={n_true['velutina']}, abeja={n_true['abeja']}, crabro={n_true['crabro']})")
    print(f"Top1 global: {top1_global:.3f} "
          f"({sum(1 for t, p in zip(true_names, pred_names) if t == p)}/{n})")
    print(f"Top1 abeja: {top1['abeja']:.3f} ({conf[('abeja','abeja')]}/{n_true['abeja']}) | "
          f"Top1 crabro: {top1['crabro']:.3f} ({conf[('crabro','crabro')]}/{n_true['crabro']}) | "
          f"Recall velutina: {recall_vel:.3f} ({conf[('velutina','velutina')]}/{n_true['velutina']})")
    print(f"Falsas alarmas: abeja->velutina {fp_abeja}/{n_true['abeja']}, "
          f"crabro->velutina {fp_crabro}/{n_true['crabro']}.")
    print("")
    print("Matriz de confusion (filas = verdad, columnas = prediccion):")
    hdr = f"{'verdad/pred':>12}" + "".join(f"{c:>10}" for c in classes)
    print(hdr)
    for t in classes:
        print(f"{t:>12}" + "".join(f"{conf[(t, p)]:>10d}" for p in classes))
    if extras:
        print("Predicciones a clases extra del modelo (p. ej. v2 con vespula/bombus):")
        for (t, p), c in sorted(extras.items()):
            print(f"  verdad={t} predicho={p}: {c}")

    if args.umbral is not None:
        u = float(args.umbral)
        iv = order.index("velutina")
        pv = [float(p[iv]) for p in prob_list]
        prom = [x >= u for x in pv]
        n_vel = sum(1 for t in true_names if t == "velutina")
        rec_u = sum(1 for t, m in zip(true_names, prom) if t == "velutina" and m) / max(1, n_vel)
        fp_u = sum(1 for t, m in zip(true_names, prom) if t != "velutina" and m)
        print("")
        print(f"Umbral Pv>={u:.3f} (promote solo si Pv>=umbral):")
        print(f"  Recall velutina@{u:.3f}: {rec_u:.3f} "
              f"({sum(1 for t, m in zip(true_names, prom) if t == 'velutina' and m)}/{n_vel})")
        print(f"  FP total@{u:.3f}: {fp_u} "
              f"(abeja+crabro promovidos por error, de {n - n_vel} negativos)")
        print("  Como leerlo: con el umbral de operating_point.json del cuaderno v2, "
              "aqui ves cuantas pilla y si toca alguna abeja en tus fotos locales.")

    colab = parse_colab_txt(COLAB_TXT)
    if colab:
        print("")
        print("Comparacion con Colab (DESCARGAS/resultados.txt):")
        for k in ("top1_global", "top1_abeja", "top1_crabro",
                  "recall_velutina", "FP_abeja_como_velutina", "FP_crabro_como_velutina"):
            print(f"  {k}: Colab={colab.get(k, '?')}")
        print(f"  Local: top1_global={top1_global:.4f} top1_abeja={top1['abeja']:.4f} "
              f"top1_crabro={top1['crabro']:.4f} recall_velutina={recall_vel:.4f} "
              f"FP_abeja_como_velutina={fp_abeja} FP_crabro_como_velutina={fp_crabro}")
        print("  Como leerlo: si los numeros locales se parecen a los de Colab, "
              "el peso viajo bien y no se rompio nada al descargarlo.")
    print("")
    print("Comparacion con el logistico local (docs/REAL_MODEL.md, validacion 76 fotos):")
    print("  Logistico: recall 0.700 (21/30) a umbral 0.5; con cero falsas alarmas "
          "(umbral 0.990) recall 0.133 (4/30). Mira solo color+densidad de bordes.")
    print(f"  YOLO (top1 directo, {n} fotos): recall {recall_vel:.3f}, "
          f"FP abeja {fp_abeja}/{n_true['abeja']}, FP crabro {fp_crabro}/{n_true['crabro']}. "
          f"Ve formas (red neuronal), asi que detecta mas velutinas, pero ojo: "
          f"el top1 no es un umbral de seguridad; para disparo real habria que calibrar "
          f"un umbral de Pv alto como se hizo con el logistico.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
