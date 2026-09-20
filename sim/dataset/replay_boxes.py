"""Replay Hornet3000+ por caja contra YOLO v1 y v2 (DIAGNÓSTICO, sin certificación).

Uso:
    python sim/dataset/replay_boxes.py [--split val] [--limit N]
        [--weights jetson/ai/yolov8n_velutina.pt]
        [--weights jetson/ai/yolov8s_velutina_v2.pt]
        [--data data/real/hornet3000]

Sin --weights se evalúan AMBOS por defecto (v1 y v2, flag repetible).

Layout real verificado (16-sep-2026):
    data/real/hornet3000/data3000/data/{train,val}/{images,labels}/
        /{Vespa_crabro,Vespa_velutina,Vespula_sp}/
    3088 imágenes train + 297 val; labels YOLO `cls cx cy w h` con
    clases 0=velutina 1=crabro 2=vulgaris (config.yaml); en train hay
    imágenes Vespula_sp sin label (= fondo, se omiten).
El recolector también acepta otros layouts (usa cualquier *.txt bajo
un directorio `labels/` del split pedido y busca su imagen gemela
cambiando labels->images y probando extensiones).

Por cada bbox GT se recorta un cuadrado de 320px centrado en la caja
(lado = max(w_px, h_px, 40px), clamp a bordes con relleno negro,
reescalado a 320x320x3) y se pasa por `predict_proba` de cada modelo:

    v1 (yolov8n_velutina.pt, 3 clases): gate Pv >= 0.9994
    v2 (yolov8s_velutina_v2.pt, 7 clases): gate Pv >= 0.9958

Umbrales = mejor punto con FP=0 calibrado sobre GBIF-380 local
(documentado en docs/REAL_MODEL.md). Mapeo: velutina -> promote,
crabro + vulgaris/vespula -> no-promote.

NO toca jetson/ai/yolo_backend.py ni detector.py ni thresholds.yaml:
carga los pesos directamente con ultralytics (solo diagnóstico).
Etiqueta: DIAGNÓSTICO.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DEFAULT = REPO_ROOT / "data" / "real" / "hornet3000"
CONFIG_DEFAULT = DATA_DEFAULT / "config.yaml"
V1_PATH = REPO_ROOT / "jetson" / "ai" / "yolov8n_velutina.pt"
V2_PATH = REPO_ROOT / "jetson" / "ai" / "yolov8s_velutina_v2.pt"
BOX_THRESHOLDS_JSON = REPO_ROOT / "jetson" / "ai" / "box_thresholds.json"

# Mejor umbral con FP=0 calibrado en GBIF-380 local (docs/REAL_MODEL.md).
THR_V1 = 0.9994
THR_V2 = 0.9958

# Barrido para recalibrar en cajas Hornet3000 val.
SWEEP_THRESHOLDS = [0.5, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999]

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Fallback si no hay config.yaml (ids según Hornet3000+ config.yaml).
FALLBACK_NAMES = {0: "Vespa_velutina", 1: "Vespa_crabro", 2: "Vespula_vulgaris"}


def default_threshold_for(weights: Path) -> float:
    """Umbral operativo según modelo: v2 -> 0.9958, resto -> 0.9994."""
    stem = Path(str(weights)).stem.lower()
    if "v2" in stem:
        return THR_V2
    return THR_V1


def parse_yolo_label(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Parsea un .txt YOLO: cada línea `cls cx cy w h` normalizados."""
    boxes: list[tuple[int, float, float, float, float]] = []
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


def load_class_names(data_root: Path) -> dict[int, str]:
    """Nombres de clase desde config.yaml/data.yaml (names) o fallback."""
    for cand in (Path(data_root) / "config.yaml",
                 Path(data_root) / "data.yaml",
                 REPO_ROOT / "data" / "real" / "hornet3000" / "config.yaml"):
        try:
            if not cand.exists():
                continue
            import yaml  # import local: solo se necesita aquí
            with open(cand, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            if isinstance(data, dict) and isinstance(data.get("names"), dict):
                return {int(k): str(v) for k, v in data["names"].items()}
            if isinstance(data, dict) and isinstance(data.get("names"), list):
                return {i: str(n) for i, n in enumerate(data["names"])}
        except Exception:
            continue
    return dict(FALLBACK_NAMES)


def expects_promote(class_name: str, cls_id: int = -1) -> bool:
    """True si la caja GT es velutina (espera promote). Resto: no-promote."""
    name = (class_name or "").lower()
    if "velutina" in name:
        return True
    if cls_id == 0 and not name:
        return True
    return False


def class_bucket(class_name: str, cls_id: int) -> str:
    """Agrupa la caja GT en velutina | crabro | vulgaris | otra."""
    name = (class_name or "").lower()
    if "velutina" in name or (cls_id == 0 and not name):
        return "velutina"
    if "crabro" in name or cls_id == 1:
        return "crabro"
    if "vulgar" in name or "vespula" in name or cls_id == 2:
        return "vulgaris"
    return "otra"


def _image_for_label(label_path: Path) -> Path | None:
    """Imagen gemela de un label: labels->images + prueba de extensiones."""
    parts = list(label_path.parts)
    if "labels" in parts:
        mirror = Path(*[p if p != "labels" else "images" for p in parts])
    else:
        mirror = label_path
    for ext in IMAGE_EXTS:
        cand = mirror.with_suffix(ext)
        if cand.exists():
            return cand
        cand_up = mirror.with_suffix(ext.upper())
        if cand_up.exists():
            return cand_up
    return None


def collect_boxes(data_root: Path, split: str,
                  limit: int = 0) -> list[tuple[Path, int, float, float, float, float]]:
    """Lista ordenada de (img_path, cls, cx, cy, w, h) hasta `limit` cajas.

    Busca *.txt bajo `labels/` del split pedido (acepta subcarpetas por
    clase). Labels vacíos o sin imagen gemela se omiten (fondo).
    `limit` = máximo de cajas (0 = todas).
    """
    data_root = Path(data_root)
    label_files = sorted(
        p for p in data_root.rglob("*.txt")
        if split in p.parts and "labels" in p.parts
    )
    items: list[tuple[Path, int, float, float, float, float]] = []
    for lab in label_files:
        boxes = parse_yolo_label(lab)
        if not boxes:
            continue
        img = _image_for_label(lab)
        if img is None:
            print(f"Aviso: {lab.name} sin imagen gemela, se omite.",
                  file=sys.stderr)
            continue
        for (cls, cx, cy, w, h) in boxes:
            items.append((img, cls, cx, cy, w, h))
            if limit and len(items) >= limit:
                return items
    return items


def load_image_rgb(img_path: Path) -> np.ndarray:
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


def make_square_crop_320(img_rgb: np.ndarray,
                         cx_n: float, cy_n: float,
                         w_n: float, h_n: float) -> np.ndarray:
    """Recorte cuadrado 320 centrado en la caja (relleno negro + resize).

    Lado = max(w_px, h_px, 40px); clamp a bordes con relleno negro y
    reescalado a 320x320x3 uint8 con cv2.resize (fallback Pillow).
    """
    img = np.asarray(img_rgb)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    img = np.ascontiguousarray(img, dtype=np.uint8)
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
        out = cv2.resize(canvas, (320, 320), interpolation=cv2.INTER_LINEAR)
        return np.ascontiguousarray(out, dtype=np.uint8)
    except ImportError:
        from PIL import Image
        return np.asarray(Image.fromarray(canvas).resize((320, 320),
                                                         Image.BILINEAR),
                          dtype=np.uint8)


def _to_224(crop_320: np.ndarray) -> np.ndarray:
    img = np.ascontiguousarray(crop_320, dtype=np.uint8)
    if img.shape != (320, 320, 3):
        raise ValueError(f"crop 320x320x3 requerido, llego {img.shape}")
    try:
        import cv2
        return np.ascontiguousarray(
            cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR))
    except ImportError:
        from PIL import Image
        return np.asarray(Image.fromarray(img).resize((224, 224),
                                                      Image.BILINEAR))


def load_cls_model(weights: Path) -> dict:
    """Carga un YOLO-cls y localiza el índice de 'velutina'."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Falta ultralytics para el replay. "
            "Instala con: pip install ultralytics (gratis)."
        ) from exc
    w = Path(str(weights))
    if not w.is_file():
        raise FileNotFoundError(f"No existe el peso {w}.")
    model = YOLO(str(w))
    names: dict = dict(getattr(model, "names", {}))
    idx = None
    for k, v in names.items():
        if str(v).lower() == "velutina":
            idx = int(k)
            break
    if idx is None:  # fallback por subcadena, sin inventar clases
        for k, v in names.items():
            if "velutina" in str(v).lower():
                idx = int(k)
                break
    if idx is None:
        raise ValueError(f"El modelo {w} no tiene clase 'velutina' "
                         f"(names={names}).")
    return {"model": model, "idx_velutina": idx, "names": names,
            "path": str(w)}


def predict_pv_batch(model_bundle: dict,
                     crops_320: list[np.ndarray]) -> list[float]:
    """Pv (prob. softmax de velutina) por crop, en lote + fallback 1-a-1."""
    if not crops_320:
        return []
    imgs224 = [_to_224(c) for c in crops_320]
    model = model_bundle["model"]
    idx = int(model_bundle["idx_velutina"])
    try:
        results = model.predict(imgs224, verbose=False, device="cpu")
        out = []
        for r in results:
            probs = np.asarray(r.probs.data.cpu()).ravel().astype(np.float64)
            out.append(min(1.0, max(0.0, float(probs[idx]))))
        return out
    except Exception:
        out = []
        for c in imgs224:
            r = model.predict(c, verbose=False, device="cpu")[0]
            probs = np.asarray(r.probs.data.cpu()).ravel().astype(np.float64)
            out.append(min(1.0, max(0.0, float(probs[idx]))))
        return out


def run_replay(items: list[tuple[Path, int, float, float, float, float]],
               names: dict[int, str],
               predict_fn,
               threshold: float) -> dict:
    """Evalúa cajas con `predict_fn(crop320) -> Pv` y gate Pv>=threshold.

    `predict_fn` inyectable: en producción es el modelo YOLO; en tests,
    un stub rápido. Devuelve stats con recall y FP por clase.
    """
    stats: dict = {"n": 0, "threshold": float(threshold),
                   "n_velutina": 0, "n_crabro": 0, "n_vulgaris": 0,
                   "n_otra": 0, "tp": 0, "fn": 0,
                   "fp_crabro": 0, "fp_vulgaris": 0, "fp_otra": 0,
                   "promoted": 0}
    img_cache: dict[Path, np.ndarray] = {}
    for (img_path, cls, cx, cy, w, h) in items:
        cname = names.get(cls, f"clase_{cls}")
        bucket = class_bucket(cname, cls)
        if img_path not in img_cache:
            try:
                img_cache[img_path] = load_image_rgb(img_path)
            except Exception as exc:  # noqa: BLE001
                print(f"Aviso: no se pudo leer {img_path} ({exc}), se omite.",
                      file=sys.stderr)
                continue
        try:
            crop = make_square_crop_320(img_cache[img_path], cx, cy, w, h)
            pv = float(predict_fn(crop))
        except Exception as exc:  # noqa: BLE001
            print(f"Aviso: fallo al evaluar {img_path} ({exc}), se omite.",
                  file=sys.stderr)
            continue
        promoted = bool(pv >= float(threshold))
        stats["n"] += 1
        stats[f"n_{bucket}"] = stats.get(f"n_{bucket}", 0) + 1
        if bucket == "velutina":
            if promoted:
                stats["tp"] += 1
            else:
                stats["fn"] += 1
        else:
            if promoted:
                stats[f"fp_{bucket}"] = stats.get(f"fp_{bucket}", 0) + 1
        if promoted:
            stats["promoted"] += 1
    denom_r = stats["tp"] + stats["fn"]
    stats["recall"] = (stats["tp"] / denom_r) if denom_r else 0.0
    return stats


def compute_sweep_rows(valid_items: list[tuple[Path, int, float, float, float, float]],
                       names: dict[int, str],
                       pvs: list[float],
                       thresholds: list[float] | None = None) -> list[dict]:
    """Barrido de umbrales: por cada thr calcula recall velutina y FP.

    FP = crabro + vulgaris promovidos por error. Devuelve lista ordenada
    por thresholds tal cual entran (SWEEP_THRESHOLDS por defecto).
    Cada fila: {umbral, recall, fp, fp_crabro, fp_vulgaris, tp, fn, n_velutina}.
    """
    if thresholds is None:
        thresholds = list(SWEEP_THRESHOLDS)
    # Precomputar buckets para no repetir class_bucket.
    buckets = [class_bucket(names.get(int(it[1]), ""), int(it[1])) for it in valid_items]
    n_vel = sum(1 for b in buckets if b == "velutina")
    rows: list[dict] = []
    for thr in thresholds:
        tp = fn = fp_crabro = fp_vulgaris = 0
        for b, pv in zip(buckets, pvs):
            try:
                pv_f = float(pv)
            except Exception:
                continue
            # NaN nunca promociona.
            if pv_f != pv_f:  # is NaN
                is_promoted = False
            else:
                is_promoted = pv_f >= float(thr)
            if b == "velutina":
                if is_promoted:
                    tp += 1
                else:
                    fn += 1
            else:
                if is_promoted:
                    if b == "crabro":
                        fp_crabro += 1
                    elif b == "vulgaris":
                        fp_vulgaris += 1
                    else:
                        # otra -> cuenta como FP vulgaris-like pero separado
                        # para no inflar FP, contamos aparte y no sumamos a fp principal
                        # pero por simplicidad lo sumamos a fp_vulgaris si es otra negativa
                        pass
        fp = fp_crabro + fp_vulgaris
        recall = (tp / (tp + fn)) if (tp + fn) else 0.0
        rows.append({
            "umbral": float(thr),
            "recall": float(recall),
            "fp": int(fp),
            "fp_crabro": int(fp_crabro),
            "fp_vulgaris": int(fp_vulgaris),
            "tp": int(tp),
            "fn": int(fn),
            "n_velutina": int(n_vel),
        })
    return rows


def pick_best_fp0(rows: list[dict]) -> dict | None:
    """Mayor recall con FP=0; empate desempata a menor umbral."""
    cand = [r for r in rows if int(r.get("fp", 99)) == 0]
    if not cand:
        return None
    # max recall, si empate menor umbral
    best = sorted(cand, key=lambda r: (-float(r["recall"]), float(r["umbral"])))[0]
    return dict(best)


def format_sweep_table(label: str, rows: list[dict], best: dict | None) -> str:
    """Tabla en texto para un modelo. Marca la fila MEJOR FP=0 con '<-- MEJOR FP=0'."""
    lines = []
    lines.append(f"[{label}] Barrido Pv sobre cajas val (umbrales {len(rows)} puntos):")
    lines.append(f"  {'umbral':>7}  {'recall':>6}   {'FP':>3}  {'FP_crabro':>8} {'FP_vulgaris':>10}   TP/FN")
    for r in rows:
        mark = " <-- MEJOR FP=0" if (best is not None and r["umbral"] == best["umbral"] and r["recall"] == best["recall"] and r["fp"] == best["fp"]) else ""
        lines.append(
            f"  {r['umbral']:7.4f}  {r['recall']:6.3f}  {r['fp']:3d}  {r['fp_crabro']:8d} {r['fp_vulgaris']:10d}   {r['tp']:3d}/{r['fn']:3d}{mark}"
        )
    if best is not None:
        lines.append(f"  => MEJOR FP=0: umbral={best['umbral']:.4f} recall={best['recall']:.3f} FP=0 ({best['tp']}/{best['tp']+best['fn']})")
    else:
        lines.append("  => Sin umbral con FP=0 en el barrido.")
    return "\n".join(lines)


def save_box_thresholds(per_model_best: dict[str, dict | None],
                        n_cajas: int,
                        out_path: Path | None = None) -> Path:
    """Guarda jetson/ai/box_thresholds.json con {v1:{umbral,recall,fp,n}, v2:{...}, n_cajas, fecha}."""
    out = Path(out_path) if out_path is not None else BOX_THRESHOLDS_JSON
    out.parent.mkdir(parents=True, exist_ok=True)
    fecha = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
    payload: dict = {
        "n_cajas": int(n_cajas),
        "fecha": fecha,
    }
    for k in ("v1", "v2"):
        b = per_model_best.get(k)
        if b is None:
            payload[k] = {"umbral": None, "recall": 0.0, "fp": None, "n": int(n_cajas), "n_velutina": 0}
        else:
            n_vel = int(b.get("n_velutina", b.get("n", n_cajas)))
            payload[k] = {
                "umbral": float(b.get("umbral")),
                "recall": float(b.get("recall")),
                "fp": int(b.get("fp", 0)),
                "n": int(n_cajas),
                "n_velutina": n_vel,
            }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return out


def build_report_text(per_model: dict[str, dict],
                      class_counts: dict[str, int],
                      split: str, n_imgs: int) -> str:
    """Informe en español llano con tabla v1 vs v2 y veredicto."""
    lines = []
    lines.append("Informe replay por cajas Hornet3000+ (DIAGNÓSTICO, "
                 "lenguaje llano)")
    lines.append("=" * 64)
    lines.append(f"Split: {split} | imágenes con caja: {n_imgs} | "
                 f"cajas GT: {sum(class_counts.values())} "
                 f"(velutina={class_counts.get('velutina', 0)}, "
                 f"crabro={class_counts.get('crabro', 0)}, "
                 f"vulgaris={class_counts.get('vulgaris', 0)})")
    lines.append("Gate operativo por modelo (mejor punto FP=0 en GBIF-380, "
                 "docs/REAL_MODEL.md):")
    for label, st in per_model.items():
        lines.append(f"  - {label}: Pv >= {st['threshold']:.4f}")
    lines.append("")
    lines.append("Tabla v1 vs v2 sobre cajas reales:")
    hdr = (f"  {'modelo':<8}{'recall velutina':>16}"
           f"{'FP crabro':>12}{'FP vulgaris':>13}{'promovidas':>12}")
    lines.append(hdr)
    for label, st in per_model.items():
        lines.append(
            f"  {label:<8}{st['recall']:>15.3f} "
            f"({st['tp']}/{st['tp'] + st['fn']})"
            f"{st.get('fp_crabro', 0):>8}/{class_counts.get('crabro', 0):<3}"
            f"{st.get('fp_vulgaris', 0):>8}/{class_counts.get('vulgaris', 0):<3}"
            f"{st['promoted']:>9}")
    lines.append("")
    lines.append("Cómo leerlo: recall = de cada 100 velutinas, cuántas "
                 "promociona el gate; FP crabro/vulgaris = cuántas avispas "
                 "que NO son velutina se cuelan por error (ideal: 0).")
    # Veredicto: más recall manda; empate -> menos FP totales.
    order = list(per_model.items())
    if len(order) >= 2:
        (la, sa), (lb, sb) = order[0], order[1]
        fpa = sa.get("fp_crabro", 0) + sa.get("fp_vulgaris", 0)
        fpb = sb.get("fp_crabro", 0) + sb.get("fp_vulgaris", 0)
        if (sa["recall"], -fpa) >= (sb["recall"], -fpb):
            win, wrec, wfp = la, sa["recall"], fpa
            lose, lrec, lfp = lb, sb["recall"], fpb
        else:
            win, wrec, wfp = lb, sb["recall"], fpb
            lose, lrec, lfp = la, sa["recall"], fpa
        lines.append(f"Veredicto: MANDA {win} (recall {wrec:.3f} con "
                     f"{wfp} FP frente a {lose} recall {lrec:.3f} con "
                     f"{lfp} FP).")
    else:
        (label, st) = order[0]
        lines.append(f"Veredicto (un solo modelo): {label} recall "
                     f"{st['recall']:.3f}.")
    lines.append("Nota honesta: los umbrales se calibraron con FP=0 en "
                 "GBIF-380 (recorte central), no en estas cajas; aquí se "
                 "reutilizan tal cual para comparar v1 vs v2 en igualdad.")
    lines.append("DIAGNÓSTICO por cajas reales, sin valor de certificación.")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Replay Hornet3000+ por caja: v1 vs v2 (diagnóstico).")
    ap.add_argument("--split", default="val",
                    help="Split a usar: val|train (defecto val)")
    ap.add_argument("--limit", type=int, default=0,
                    help="Máximo de cajas (0 = todas)")
    ap.add_argument("--weights", action="append", default=None,
                    help="Peso YOLO-cls (repetible; por defecto v1 y v2)")
    ap.add_argument("--data", default=str(DATA_DEFAULT),
                    help="Raíz del dataset Hornet3000+")
    ap.add_argument("--sweep", action="store_true",
                    help="Barrido de umbrales Pv y guarda box_thresholds.json")
    args = ap.parse_args(argv)

    weights_list = args.weights or [str(V1_PATH), str(V2_PATH)]
    for w in weights_list:
        if not Path(w).is_file():
            print(f"ERROR: no existe el peso {w}.", file=sys.stderr)
            return 2

    data_root = Path(args.data)
    names = load_class_names(data_root)
    items = collect_boxes(data_root, args.split, args.limit)
    if not items:
        print(f"ERROR: no se encontraron cajas en {data_root} "
              f"(split={args.split}).", file=sys.stderr)
        return 2
    print(f"Cajas GT: {len(items)} en split={args.split} "
          f"(límite={args.limit or 'todas'}).", file=sys.stderr)

    # Recortes una sola vez (cache de imagen + crop por caja).
    img_cache: dict = {}
    crops: list = []
    for (img_path, cls, cx, cy, w, h) in items:
        if img_path not in img_cache:
            try:
                img_cache[img_path] = load_image_rgb(img_path)
            except Exception as exc:  # noqa: BLE001
                print(f"Aviso: no se pudo leer {img_path} ({exc}).",
                      file=sys.stderr)
                img_cache[img_path] = None
        img = img_cache[img_path]
        if img is None:
            crops.append(None)
        else:
            crops.append(make_square_crop_320(img, cx, cy, w, h))

    per_model: dict[str, dict] = {}
    sweep_rows_map: dict[str, list[dict]] = {}
    best_map: dict[str, dict | None] = {}
    for w in weights_list:
        bundle = load_cls_model(Path(w))
        p = Path(w)
        stem = p.stem.lower()
        if "v2" in stem:
            label = "v2"
        elif p == V1_PATH or stem.startswith("yolov8n"):
            label = "v1"
        else:
            label = p.stem
        thr = default_threshold_for(p)
        ok = [c is not None for c in crops]
        pvs = predict_pv_batch(bundle, [c for c in crops if c is not None])
        it = iter(pvs)
        ordered = [next(it) if good else float("nan") for good in ok]

        state = {"i": 0}
        vals = list(ordered)

        def _fn(_crop, _vals=vals, _st=state):
            v = _vals[_st["i"]]
            _st["i"] += 1
            return v

        valid = [box for box, good in zip(items, ok) if good]
        stats = run_replay(valid, names, _fn, thr)
        per_model[label] = stats
        print(f"[{label}] {p.name}: recall={stats['recall']:.3f} "
              f"({stats['tp']}/{stats['tp'] + stats['fn']}) "
              f"FP crabro={stats.get('fp_crabro', 0)} "
              f"FP vulgaris={stats.get('fp_vulgaris', 0)} "
              f"umbral={thr:.4f}", file=sys.stderr)
        if args.sweep:
            # Barrido sobre los mismos Pv ya calculados (sin re-inferir)
            rows = compute_sweep_rows(valid, names, pvs, SWEEP_THRESHOLDS)
            best = pick_best_fp0(rows)
            sweep_rows_map[label] = rows
            best_map[label] = best
            # Tabla por modelo a stdout (y también a stderr para visibilidad)
            tbl = format_sweep_table(label, rows, best)
            print(tbl)
            print(tbl, file=sys.stderr)

    class_counts = {"velutina": 0, "crabro": 0, "vulgaris": 0}
    for (_, cls, _, _, _, _) in items:
        b = class_bucket(names.get(cls, ""), cls)
        if b in class_counts:
            class_counts[b] += 1
    n_imgs = len({str(i[0]) for i in items})
    print(build_report_text(per_model, class_counts, args.split, n_imgs))
    if args.sweep:
        # Garantizar claves v1/v2 en el json aunque el usuario haya filtrado --weights
        json_best: dict[str, dict | None] = {}
        for k in ("v1", "v2"):
            json_best[k] = best_map.get(k)
        # Si se pidió un solo peso con label custom, mapearlo también para no perder info
        # pero sin romper el contrato v1/v2; si falta alguno, queda None.
        out = save_box_thresholds(json_best, len(items))
        print(f"Barrido guardado en {out} (n_cajas={len(items)})", file=sys.stderr)
        print(f"Barrido guardado en {out} (n_cajas={len(items)})")
        # Veredicto EN CAJAS con umbrales recalibrados FP=0
        if best_map:
            # comparar mejor recall FP=0 entre modelos con datos disponibles
            avail = {k: v for k, v in best_map.items() if v is not None}
            if len(avail) >= 2:
                # ordenar por recall desc, empate menor umbral
                sorted_best = sorted(avail.items(), key=lambda kv: (-kv[1]["recall"], kv[1]["umbral"]))
                win_label, win_best = sorted_best[0]
                lose_label, lose_best = sorted_best[1]
                print(f"Veredicto EN CAJAS (umbral FP=0 recalibrado): MANDA {win_label} "
                      f"(umbral {win_best['umbral']:.4f} recall {win_best['recall']:.3f}) "
                      f"frente a {lose_label} (umbral {lose_best['umbral']:.4f} recall {lose_best['recall']:.3f}).")
            elif len(avail) == 1:
                only_label, only_best = next(iter(avail.items()))
                print(f"Veredicto EN CAJAS (un solo modelo): {only_label} umbral {only_best['umbral']:.4f} recall {only_best['recall']:.3f} FP=0.")
            else:
                print("Veredicto EN CAJAS: ningún modelo alcanza FP=0 en el barrido.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
