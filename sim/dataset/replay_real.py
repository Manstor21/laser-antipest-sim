"""Replay de dataset YOLO real contra el stub sim-to-real (diagnóstico, sin certificación).

Uso:
    python sim/dataset/replay_real.py --data data/real/hornet3000 [--limit 200] [--split val]

Layout aceptado en DIR (--data):
  A) DIR/data.yaml con `names:` y rutas train/val (formato YOLO), o
  B) DIR/images/ + DIR/labels/ + DIR/classes.txt (también vale images/<split>/).

Por cada bbox YOLO (`cls cx cy w h` normalizados) se recorta un cuadrado de
320 px centrado en la caja (lado = max(w, h, 40px), clamp a bordes con relleno
negro, reescalado a 320), se pasa por `predict` + `measure_r3_r4_from_mask` y
se aplica el gate:

    Pv >= p_velutina_min AND Pb <= p_bee_max AND R3 AND R4

Umbrales desde jetson/ai/thresholds.yaml (p_velutina_min=0.995, p_bee_max=0.001).
R3/R4 con los rangos reales de jetson/fusion/hard_rules.py:
  R3: thorax_v < 60 AND 15 <= band_h <= 45 AND band_s > 80  (r3_hsv_ok)
  R4: 1.6 <= wingspan/body <= 2.4                          (r4_ratio_ok)
Sin --data o con directorio vacío: imprime ayuda de descarga y sale con exit 2.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML siempre instalado en este repo
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[2]
THRESHOLDS_PATH = REPO_ROOT / "jetson" / "ai" / "thresholds.yaml"

HELP_TEXT = """\
No se indicó dataset (--data DIR vacío o sin imágenes).

Cómo conseguir datos reales (formato YOLO: images/ + labels/ + data.yaml):

  1) Kaggle Hornet3000+ (vespa velutina vs crabro / vespulina / vulgaris):
     https://www.kaggle.com/datasets/marcoryvandijk/vespa-velutina-v-crabro-vespulina-vulgaris
     - Crea una cuenta gratis en Kaggle, abre el enlace y pulsa Download.
     - Descomprime (son cientos de MB) en data/real/hornet3000/
  2) Roboflow vespa-velutina 3715:
     https://universe.roboflow.com/vespa-velutina-pobjr/vespa-velutina-nubcn/dataset/1
     - Crea una cuenta gratis en Roboflow, abre el proyecto, pulsa Download,
       elige formato YOLOv8 y descomprime en data/real/velutina3715/
  3) Abejas (negativos) Honey Bee 909:
     https://universe.roboflow.com/bscs-kxc9w/honey-bee-detection-model-zgjnb-8fmzo/dataset/1
     - Descarga igual en formato YOLO y descomprime en data/real/abejas/

Después ejecuta, por ejemplo:
  python sim/dataset/replay_real.py --data data/real/hornet3000 --limit 200
  python sim/dataset/replay_real.py --data data/real/velutina3715 --limit 200
  python sim/dataset/replay_real.py --data data/real/abejas --limit 200

Opciones: --data DIR  --limit N (defecto 200)  --split val|train (defecto val)
"""

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Clases que esperamos que NO promocionen (negativos conocidos).
NO_PROMOTE_KEYS = ("crabro", "vulgaris", "vespula", "bee", "apis", "wasp")


def load_thresholds():
    """Lee p_velutina_min / p_bee_max de jetson/ai/thresholds.yaml."""
    pv_min, pb_max = 0.995, 0.001
    try:
        if yaml is not None and THRESHOLDS_PATH.exists():
            with open(THRESHOLDS_PATH, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            if isinstance(data, dict):
                pv_min = float(data.get("p_velutina_min", pv_min))
                pb_max = float(data.get("p_bee_max", pb_max))
    except Exception:
        pass
    return pv_min, pb_max


def expects_promote(class_name: str):
    """Mapeo por nombre (minúsculas). Devuelve (espera_promote, es_desconocida)."""
    name = (class_name or "").lower()
    if "velutina" in name:
        return True, False
    for key in NO_PROMOTE_KEYS:
        if key in name:
            return False, False
    return False, True  # desconocida -> no-promote + aviso


def parse_yolo_label(label_path):
    """Parsea un .txt YOLO: cada línea `cls cx cy w h` normalizados."""
    boxes = []
    p = Path(label_path)
    if not p.exists():
        return boxes
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
                cx, cy, w, h = (float(parts[1]), float(parts[2]),
                                float(parts[3]), float(parts[4]))
            except ValueError:
                continue
            boxes.append((cls, cx, cy, w, h))
    return boxes


def load_class_names(data_dir: Path):
    """Nombres de clase desde data.yaml (names) o classes.txt."""
    data_yaml = data_dir / "data.yaml"
    if data_yaml.exists() and yaml is not None:
        try:
            with open(data_yaml, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            if isinstance(data, dict) and "names" in data:
                names = data["names"]
                if isinstance(names, dict):  # {0: name, ...}
                    return [names[k] for k in sorted(names)]
                return [str(n) for n in names]
        except Exception:
            pass
    classes_txt = data_dir / "classes.txt"
    if classes_txt.exists():
        with open(classes_txt, "r", encoding="utf-8") as fh:
            return [ln.strip() for ln in fh if ln.strip()]
    return []


def _glob_images(folder: Path):
    found = []
    if not folder.is_dir():
        return found
    for ext in IMAGE_EXTS:
        found.extend(sorted(folder.glob(f"*{ext}")))
        found.extend(sorted(folder.glob(f"*{ext.upper()}")))
    # quitar duplicados manteniendo orden
    seen, out = set(), []
    for p in found:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return sorted(out)


def _resolve_split_dir(data_dir: Path, split: str):
    """Resuelve el directorio de imágenes para el split pedido."""
    data_yaml = data_dir / "data.yaml"
    if data_yaml.exists() and yaml is not None:
        try:
            with open(data_yaml, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            if isinstance(data, dict):
                cand = data.get(split) or data.get("val") or data.get("train")
                if cand:
                    for base in (Path(str(cand)), data_dir / str(cand)):
                        if base.is_dir():
                            return base
                        # A veces el yaml apunta a .../images/val pero solo
                        # existe .../images ; sube niveles hasta encontrar algo.
                        cur = base
                        for _ in range(3):
                            cur = cur.parent
                            if cur.is_dir() and _glob_images(cur):
                                return cur
        except Exception:
            pass
    # Layout estándar con subcarpeta de split.
    for cand in (data_dir / "images" / split,
                 data_dir / split / "images",
                 data_dir / "images",
                 data_dir):
        imgs = _glob_images(cand)
        if imgs:
            # Si pedimos un split concreto y existe su subcarpeta, respétala.
            sub = data_dir / "images" / split
            if split and sub.is_dir() and _glob_images(sub):
                return sub
            return cand
    return None


def find_label_for_image(img_path: Path, data_dir: Path, split: str):
    """Busca el .txt YOLO gemelo de una imagen."""
    stem = img_path.stem
    candidates = []
    parts = list(img_path.parts)
    if "images" in parts:  # espejo images/ -> labels/
        mirror = Path(*[p if p != "images" else "labels" for p in parts])
        candidates.append(mirror.with_suffix(".txt"))
    for cand in (data_dir / "labels" / split / f"{stem}.txt",
                 data_dir / "labels" / f"{stem}.txt",
                 data_dir / split / "labels" / f"{stem}.txt",
                 img_path.with_suffix(".txt")):
        candidates.append(cand)
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def collect_boxes(data_dir: Path, split: str, limit: int):
    """Lista ordenada de (img_path, cls, cx, cy, w, h) hasta `limit` cajas."""
    img_dir = _resolve_split_dir(data_dir, split)
    if img_dir is None:
        return []
    items = []
    for img_path in _glob_images(img_dir):
        label_path = find_label_for_image(img_path, data_dir, split)
        if label_path is None:
            continue
        for (cls, cx, cy, w, h) in parse_yolo_label(label_path):
            items.append((img_path, cls, cx, cy, w, h))
            if limit and len(items) >= limit:
                return items
    return items


def load_image_rgb(img_path: Path):
    """Lee imagen a RGB uint8 HxWx3 (cv2.imread, fallback Pillow)."""
    try:
        import cv2  # import local: respeta fallback Pillow
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is not None:
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except ImportError:
        pass
    except Exception:
        pass
    from PIL import Image
    with Image.open(img_path) as im:
        return np.asarray(im.convert("RGB"))


def make_square_crop_320(img_rgb, cx_n, cy_n, w_n, h_n):
    """Recorte cuadrado 320 centrado en la caja (relleno negro + resize).

    Lado = max(w_px, h_px, 40px); clamp a bordes con relleno negro y
    reescalado a 320x320x3 con cv2.resize (fallback Pillow).
    """
    img = np.asarray(img_rgb)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    H, W = img.shape[:2]
    cx, cy = float(cx_n) * W, float(cy_n) * H
    side = max(float(w_n) * W, float(h_n) * H, 40.0)
    side = max(int(round(side)), 1)
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    sx1 = min(W, x0 + side) - max(0, x0)
    sy1 = min(H, y0 + side) - max(0, y0)
    if sx1 > 0 and sy1 > 0:
        canvas[sy0:sy0 + sy1, sx0:sx0 + sx1] = img[dy0:dy0 + sy1, dx0:dx0 + sx1]
    try:
        import cv2
        return cv2.resize(canvas, (320, 320), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        from PIL import Image
        return np.asarray(Image.fromarray(canvas).resize((320, 320),
                                                         Image.BILINEAR))


def evaluate_crop(crop, pv_min, pb_max):
    """Pasa un crop 320 por predict + medida R3/R4 y aplica el gate."""
    from jetson.ai.detector import predict
    from jetson.fusion.fusion_pipeline import measure_r3_r4_from_mask
    from jetson.fusion.hard_rules import r3_hsv_ok, r4_ratio_ok

    out = predict(np.ascontiguousarray(crop, dtype=np.uint8))
    pv = float(out["p_velutina"])
    pb = float(out["p_bee"])
    feats = measure_r3_r4_from_mask(out["mask"])
    r3 = bool(r3_hsv_ok(feats["thorax_v"], feats["band_h"], feats["band_s"]))
    r4 = bool(r4_ratio_ok(feats["wingspan"], feats["body"]))
    promoted = bool(pv >= pv_min and pb <= pb_max and r3 and r4)
    return {"p_velutina": pv, "p_bee": pb, "feats": dict(feats),
            "r3": r3, "r4": r4, "promoted": promoted}


def build_report_text(stats: dict) -> str:
    lines = []
    lines.append("Informe replay sim-to-real (diágnostico, lenguaje llano)")
    lines.append("=" * 60)
    lines.append(f"Dataset: {stats['data_dir']}  (split={stats['split']}, "
                 f"límite={stats['limit']})")
    lines.append(f"Umbrales IA: Pv>={stats['pv_min']:.3f} y "
                 f"Pb<={stats['pb_max']:.3f}; R3/R4 según hard_rules.py")
    lines.append(f"Cajas procesadas: {stats['n']} "
                 f"({stats['n']} procesadas en total)")
    lines.append(f"Esperadas velutina: {stats['n_vel']} | "
                 f"esperadas no-velutina: {stats['n_other']}")
    lines.append(f"Promocionadas (gate completo): {stats['n_promoted']}")
    lines.append(f"recall velutina (de cada 100 velutinas, "
                 f"cuántas detectamos): {stats['recall']:.3f} "
                 f"({stats['tp']}/{stats['tp'] + stats['fn']})")
    lines.append(f"precisión (de cada 100 alarmas, cuántas eran "
                 f"velutina de verdad): {stats['precision']:.3f} "
                 f"({stats['tp']}/{stats['tp'] + stats['fp_total']})")
    lines.append("Falsos positivos (FP) por clase (no-velutina promocionada):")
    if stats["fp_by_class"]:
        for name in sorted(stats["fp_by_class"]):
            lines.append(f"  - {name}: {stats['fp_by_class'][name]}")
    else:
        lines.append("  - ninguno")
    if stats["unknown_classes"]:
        lines.append("Aviso: clases desconocidas tratadas como no-promote: "
                     + ", ".join(sorted(stats["unknown_classes"])))
    lines.append("")
    lines.append("Cómo leerlo: recall alto = detectamos casi todas las "
                 "velutinas; FP por clase = qué otras avispas/abejas nos "
                 "confunden; precisión baja = muchas alarmas falsas.")
    lines.append("Con el stub actual, Pv=0.10+0.85*fg y Pb=0.30*(1-fg)+0.02*fg, "
                 "así que solo promocionan recortes muy brillantes "
                 "(fg cercano a 1) que además pasen R3/R4: espera recall bajo "
                 "y pocos FP; es normal, el stub no es el modelo final.")
    lines.append("DIAGNÓSTICO SIM-TO-REAL, sin valor de certificación.")
    return "\n".join(lines) + "\n"


def run_replay(data_dir: Path, limit: int, split: str) -> dict:
    pv_min, pb_max = load_thresholds()
    names = load_class_names(data_dir)
    boxes = collect_boxes(data_dir, split, limit)
    stats = {"data_dir": str(data_dir), "split": split, "limit": limit,
             "pv_min": pv_min, "pb_max": pb_max, "n": 0, "n_vel": 0,
             "n_other": 0, "tp": 0, "fn": 0, "fp_total": 0,
             "n_promoted": 0, "fp_by_class": {}, "unknown_classes": set()}
    img_cache = {}
    for (img_path, cls, cx, cy, w, h) in boxes:
        cname = names[cls] if 0 <= cls < len(names) else f"clase_{cls}"
        expected, unknown = expects_promote(cname)
        if unknown:
            stats["unknown_classes"].add(str(cname))
            print(f"Aviso: clase desconocida '{cname}' -> se trata como "
                  f"no-promote.", file=sys.stderr)
        if img_path not in img_cache:
            try:
                img_cache[img_path] = load_image_rgb(img_path)
            except Exception as exc:
                print(f"Aviso: no se pudo leer {img_path} ({exc}), se omite.",
                      file=sys.stderr)
                continue
        try:
            crop = make_square_crop_320(img_cache[img_path], cx, cy, w, h)
            res = evaluate_crop(crop, pv_min, pb_max)
        except Exception as exc:
            print(f"Aviso: fallo al evaluar {img_path} ({exc}), se omite.",
                  file=sys.stderr)
            continue
        stats["n"] += 1
        if expected:
            stats["n_vel"] += 1
            if res["promoted"]:
                stats["tp"] += 1
            else:
                stats["fn"] += 1
        else:
            stats["n_other"] += 1
            if res["promoted"]:
                stats["fp_total"] += 1
                stats["fp_by_class"][str(cname)] = \
                    stats["fp_by_class"].get(str(cname), 0) + 1
        if res["promoted"]:
            stats["n_promoted"] += 1
    denom_r = stats["tp"] + stats["fn"]
    stats["recall"] = (stats["tp"] / denom_r) if denom_r else 0.0
    denom_p = stats["tp"] + stats["fp_total"]
    stats["precision"] = (stats["tp"] / denom_p) if denom_p else 1.0
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Replay YOLO real contra stub sim-to-real (diagnóstico).")
    ap.add_argument("--data", default=None,
                    help="Carpeta del dataset (con data.yaml o images/+labels/)")
    ap.add_argument("--limit", type=int, default=200,
                    help="Máximo de cajas a evaluar (defecto 200)")
    ap.add_argument("--split", default="val",
                    help="Split a usar: val|train (defecto val)")
    args = ap.parse_args(argv)
    if not args.data:
        print(HELP_TEXT)
        raise SystemExit(2)
    data_dir = Path(args.data)
    if not data_dir.is_dir() or not collect_boxes(data_dir, args.split,
                                                  limit=1):
        if data_dir.is_dir():
            print(f"No se encontraron imágenes/cajas en {data_dir} "
                  f"(split={args.split}).\n")
        print(HELP_TEXT)
        raise SystemExit(2)
    stats = run_replay(data_dir, args.limit, args.split)
    if stats["n"] == 0:
        print(f"No se pudo procesar ninguna caja en {data_dir}.\n")
        print(HELP_TEXT)
        raise SystemExit(2)
    print(build_report_text(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
