"""Rejuego de fotos reales GBIF/iNat (sin bboxes) contra el stub sim-to-real.

Uso:
    python sim/dataset/replay_gbif.py [--dirs data/real/gbif_velutina data/real/gbif_abeja ...] [--limit 150]

Por imagen se toma un recorte CENTRAL cuadrado de 320 px (sin bboxes):
  - Si la imagen es mayor de 320 en ambos lados: crop central 320x320.
  - Si es menor: se reescala el lado mayor a 320 y luego crop central,
    con relleno negro si hace falta para llegar a 320x320x3.
El crop pasa por `jetson.ai.detector.predict` + `measure_r3_r4_from_mask`
y el gate de thresholds.yaml:
    Pv >= p_velutina_min AND Pb <= p_bee_max AND R3 AND R4
(R3/R4 con los rangos reales de jetson/fusion/hard_rules.py).

Mapeo de carpetas (por nombre):
    *gbif_velutina* -> espera promote (positivo)
    *gbif_abeja* / *gbif_crabro* -> espera no-promote (negativos)

Sin --dirs válidos (o sin imágenes): imprime ayuda y sale con exit 2.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
THRESHOLDS_PATH = REPO_ROOT / "jetson" / "ai" / "thresholds.yaml"

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

DEFAULT_DIRNAMES = ("gbif_velutina", "gbif_abeja", "gbif_crabro")

HELP_TEXT = """\
No se indicaron carpetas con fotos (--dirs vacío o sin imágenes).

Cómo conseguir fotos reales libres (sin cuentas):
  python sim/dataset/fetch_gbif.py
  - Descarga fotos CC de GBIF (fallback iNaturalist) en:
      data/real/gbif_velutina/  (Vespa velutina, espera promote)
      data/real/gbif_abeja/     (Apis mellifera, espera no-promote)
      data/real/gbif_crabro/    (Vespa crabro, espera no-promote)

Después ejecuta, por ejemplo:
  python sim/dataset/replay_gbif.py --limit 150
  python sim/dataset/replay_gbif.py --dirs data/real/gbif_velutina data/real/gbif_abeja --limit 50

Opciones: --dirs DIR [DIR ...]  --limit N por carpeta (defecto 200)
"""


def load_thresholds():
    try:
        from jetson.ai.temporal import get_thresholds
        thresholds = get_thresholds()
        pv_min = float(thresholds.get("p_velutina_min", 0.995))
        pb_max = float(thresholds.get("p_bee_max", 0.001))
        return pv_min, pb_max
    except Exception:
        return 0.995, 0.001


def expects_promote_for_dir(dirname: str) -> tuple[bool, str]:
    """(espera_promote, etiqueta) según el nombre de la carpeta."""
    low = (dirname or "").lower()
    if "velutina" in low:
        return True, "Vespa velutina"
    if "abeja" in low or "mellifera" in low or "apis" in low:
        return False, "Apis mellifera"
    if "crabro" in low:
        return False, "Vespa crabro"
    return False, dirname or "desconocida"


def load_image_rgb(img_path: Path):
    try:
        import cv2  # import local

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


def _resize_rgb(img: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    try:
        import cv2

        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        from PIL import Image

        return np.asarray(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))


def make_center_crop_320(img_rgb) -> np.ndarray:
    """Recorte central cuadrado 320x320x3.

    - Si H>=320 y W>=320: crop central directo.
    - Si es menor: reescala el lado mayor a 320 (o el menor a 320 cuando solo
      un lado es pequeño, para cubrir), luego crop central; relleno negro si
      hace falta.
    """
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
    if max(H, W) < 320:
        scale = 320.0 / max(H, W)
    else:
        scale = 320.0 / min(H, W)
    new_w = max(1, int(round(W * scale)))
    new_h = max(1, int(round(H * scale)))
    img = _resize_rgb(img, new_w, new_h)
    H, W = img.shape[:2]
    canvas = np.zeros((320, 320, 3), dtype=np.uint8)
    y0, x0 = (H - 320) // 2, (W - 320) // 2
    # Región de origen (clamp) y de destino.
    sy0, sy1 = max(0, y0), min(H, y0 + 320)
    sx0, sx1 = max(0, x0), min(W, x0 + 320)
    dy0 = sy0 - y0 if y0 < 0 else 0
    dx0 = sx0 - x0 if x0 < 0 else 0
    if sy1 > sy0 and sx1 > sx0:
        canvas[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = img[sy0:sy1, sx0:sx1]
    return np.ascontiguousarray(canvas)


def evaluate_crop(crop, pv_min, pb_max):
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


def glob_images(folder: Path):
    found: list[Path] = []
    if not folder.is_dir():
        return found
    for ext in IMAGE_EXTS:
        found.extend(sorted(folder.glob(f"*{ext}")))
        found.extend(sorted(folder.glob(f"*{ext.upper()}")))
    seen, out = set(), []
    for p in found:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return sorted(out)


def run_replay(dirs: list[Path], limit: int) -> dict:
    pv_min, pb_max = load_thresholds()
    stats = {
        "pv_min": pv_min, "pb_max": pb_max, "limit": limit,
        "n_total": 0, "n_promoted": 0,
        "per_species": {},  # etiqueta -> {n, promoted, espera}
        "tp": 0, "fn": 0, "fp_total": 0,
        "fp_abeja": 0, "fp_crabro": 0,
        "recall": 0.0, "precision": 1.0,
        "dirs": [str(d) for d in dirs],
    }
    for d in dirs:
        espera, etiqueta = expects_promote_for_dir(Path(d).name)
        entry = stats["per_species"].setdefault(etiqueta, {"n": 0, "promoted": 0, "espera": espera})
        for img_path in glob_images(Path(d))[:limit if limit else None]:
            try:
                img = load_image_rgb(img_path)
            except Exception as exc:  # noqa: BLE001
                print(f"Aviso: no se pudo leer {img_path} ({exc}), se omite.", file=sys.stderr)
                continue
            try:
                crop = make_center_crop_320(img)
                res = evaluate_crop(crop, pv_min, pb_max)
            except Exception as exc:  # noqa: BLE001
                print(f"Aviso: fallo al evaluar {img_path} ({exc}), se omite.", file=sys.stderr)
                continue
            stats["n_total"] += 1
            entry["n"] += 1
            if res["promoted"]:
                stats["n_promoted"] += 1
                entry["promoted"] += 1
            if espera:
                if res["promoted"]:
                    stats["tp"] += 1
                else:
                    stats["fn"] += 1
            else:
                if res["promoted"]:
                    stats["fp_total"] += 1
                    low = etiqueta.lower()
                    if "mellifera" in low or "apis" in low or "abeja" in low:
                        stats["fp_abeja"] += 1
                    elif "crabro" in low:
                        stats["fp_crabro"] += 1
    denom_r = stats["tp"] + stats["fn"]
    stats["recall"] = (stats["tp"] / denom_r) if denom_r else 0.0
    denom_p = stats["tp"] + stats["fp_total"]
    stats["precision"] = (stats["tp"] / denom_p) if denom_p else 1.0
    return stats


def build_report_text(stats: dict) -> str:
    lines = []
    lines.append("Informe replay GBIF sim-to-real (diagnóstico, lenguaje llano)")
    lines.append("=" * 60)
    lines.append(f"Carpetas: {', '.join(stats['dirs'])}  (límite por carpeta={stats['limit']})")
    lines.append(f"Umbrales IA: Pv>={stats['pv_min']:.3f} y Pb<={stats['pb_max']:.3f}; R3/R4 según hard_rules.py")
    lines.append(f"Imágenes procesadas en total: {stats['n_total']}")
    for name in sorted(stats["per_species"]):
        e = stats["per_species"][name]
        lines.append(f"  - {name}: N={e['n']}, promocionadas={e['promoted']} "
                     f"(espera={'promote' if e['espera'] else 'no-promote'})")
    vel = stats["per_species"].get("Vespa velutina", {"n": 0, "promoted": 0})
    lines.append(f"recall velutina (de cada 100 velutinas, cuántas detectamos): "
                 f"{stats['recall']:.3f} ({stats['tp']}/{stats['tp'] + stats['fn']})")
    n_abeja = stats["per_species"].get("Apis mellifera", {"n": 0})["n"]
    n_crabro = stats["per_species"].get("Vespa crabro", {"n": 0})["n"]
    lines.append(f"FP abeja (abejas confundidas con velutina): {stats['fp_abeja']}/{n_abeja}")
    lines.append(f"FP crabro (crabros confundidos con velutina): {stats['fp_crabro']}/{n_crabro}")
    lines.append(f"precisión (de cada 100 alarmas, cuántas eran velutina de verdad): "
                 f"{stats['precision']:.3f} ({stats['tp']}/{stats['tp'] + stats['fp_total']})")
    lines.append("")
    lines.append("Cómo leerlo: recall alto = detectamos casi todas las velutinas; "
                 "FP = qué otros insectos nos confunden; precisión baja = muchas alarmas falsas.")
    lines.append("Nota honesta: el recorte es el CENTRO de la foto (320x320, sin bboxes), "
                 "así que el insecto puede quedar fuera del recorte o salir cortado; "
                 "eso mete ruido de etiquetado y puede bajar el recall o subir los FP "
                 "aunque el detector funcione bien con el insecto centrado.")
    lines.append("Con el stub actual, Pv=0.10+0.85*fg y Pb=0.30*(1-fg)+0.02*fg, así que "
                 "solo promocionan recortes muy brillantes que además pasen R3/R4: "
                 "espera recall bajo y pocos FP; es normal, el stub no es el modelo final.")
    lines.append("DIAGNÓSTICO SIM-TO-REAL, sin valor de certificación.")
    return "\n".join(lines) + "\n"


def resolve_default_dirs() -> list[Path]:
    base = REPO_ROOT / "data" / "real"
    return [base / name for name in DEFAULT_DIRNAMES]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Replay GBIF sin bboxes (crop central 320, diagnóstico).")
    ap.add_argument("--dirs", nargs="*", default=None,
                    help="Carpetas con fotos (defecto: data/real/gbif_*/)")
    ap.add_argument("--limit", type=int, default=200,
                    help="Máximo de imágenes por carpeta (defecto 200)")
    args = ap.parse_args(argv)
    dirs = [Path(d) for d in args.dirs] if args.dirs else resolve_default_dirs()
    valid = [d for d in dirs if d.is_dir() and glob_images(d)]
    if not valid:
        print(f"No se encontraron imágenes en: {', '.join(str(d) for d in dirs)}\n")
        print(HELP_TEXT)
        raise SystemExit(2)
    if len(valid) < len(dirs):
        missing = [str(d) for d in dirs if d not in valid]
        print(f"Aviso: sin imágenes en: {', '.join(missing)} (se sigue con el resto).",
              file=sys.stderr)
    stats = run_replay(valid, args.limit)
    if stats["n_total"] == 0:
        print("No se pudo procesar ninguna imagen.\n")
        print(HELP_TEXT)
        raise SystemExit(2)
    print(build_report_text(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
