"""Prepara dataset Hornet3000+ para YOLO detección (plano, sin subcarpetas).

Origen: data/real/hornet3000/data3000/data/{train,val}/{images,labels}/Vespa_*
Destino: data/det/{train,val}/{images,labels}/  (plano, sin subcarpetas)

- Lista real del disco (no asume): cuenta imágenes train/val y labels.
- Aplana: ultralytics espera train/images/*.jpg y train/labels/*.txt sin subcarpetas.
  Copia cada imagen encontrada recursivamente a destino plano; si su .txt
  existe (images->labels, mismo stem) lo copia también; si no, la imagen
  queda como fondo (sin label, ultralytics la trata como background).
- No duplica si ya existe y tiene mismo tamaño.
- Genera data/det/data.yaml con path absoluto y names correctos.

Uso:
    python sim/dataset/prepare_hornet.py
    python sim/dataset/prepare_hornet.py --check   # solo lista y verifica
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_BASE = REPO_ROOT / "data" / "real" / "hornet3000" / "data3000" / "data"
DST_BASE = REPO_ROOT / "data" / "det"
DATA_YAML = DST_BASE / "data.yaml"

NAMES = ["vespa_velutina", "vespa_crabro", "vespula_vulgaris"]
# clase 0=velutina 1=crabro 2=vulgaris según config.yaml original

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_real(base: Path = SRC_BASE):
    stats = {}
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            p = base / split / kind
            if not p.exists():
                stats[(split, kind)] = {"flat": 0, "recursive": 0, "subfolders": []}
                continue
            subfolders = [d.name for d in p.iterdir() if d.is_dir()]
            flat = sum(1 for f in p.iterdir() if f.is_file())
            rec = len([f for f in p.rglob("*") if f.is_file()])
            stats[(split, kind)] = {"flat": flat, "recursive": rec, "subfolders": subfolders}
    return stats


def find_images(src_split_images: Path):
    # recursivo, filtra por extensión de imagen
    out = []
    for p in src_split_images.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            out.append(p)
    return sorted(out)


def find_label_for_image(img_path: Path, src_split_labels: Path):
    """Busca .txt gemelo: primero por subcarpeta espejo, luego glob por stem."""
    stem = img_path.stem
    # intenta espejo: src/images/Vespa_velutina/a.jpg -> src/labels/Vespa_velutina/a.txt
    # calculamos relativo a src_split_images
    try:
        rel = img_path.relative_to(src_split_labels.parent / "images")
        # rel incluye subcarpeta + filename
        candidate = src_split_labels / rel.with_suffix(".txt")
        if candidate.exists():
            return candidate
    except Exception:
        pass
    # fallback: busca cualquier *.txt con mismo stem bajo labels
    matches = list(src_split_labels.rglob(f"{stem}.txt"))
    if matches:
        return matches[0]
    return None


def prepare(force: bool = False):
    print(f"[prepare_hornet] SRC={SRC_BASE}")
    print(f"[prepare_hornet] DST={DST_BASE}")
    if not SRC_BASE.exists():
        print(f"ERROR: no existe SRC {SRC_BASE}", file=sys.stderr)
        return 1
    # listado real
    stats = count_real(SRC_BASE)
    for (split, kind), v in stats.items():
        print(f"  {split}/{kind}: subfolders={v['subfolders']} flat={v['flat']} recursive={v['recursive']}")
    # prepara destinos
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            (DST_BASE / split / kind).mkdir(parents=True, exist_ok=True)

    total_copied = {"train": 0, "val": 0}
    total_labels = {"train": 0, "val": 0}
    total_bg = {"train": 0, "val": 0}

    for split in ("train", "val"):
        src_img_dir = SRC_BASE / split / "images"
        src_lbl_dir = SRC_BASE / split / "labels"
        dst_img_dir = DST_BASE / split / "images"
        dst_lbl_dir = DST_BASE / split / "labels"

        images = find_images(src_img_dir)
        print(f"[{split}] imágenes encontradas recursivas: {len(images)}")
        for img in images:
            dst_img = dst_img_dir / img.name
            # sin duplicar si ya plano
            if dst_img.exists() and not force:
                if dst_img.stat().st_size == img.stat().st_size:
                    # asume igual, no copia
                    pass
                else:
                    shutil.copy2(img, dst_img)
                    total_copied[split] += 1
            else:
                shutil.copy2(img, dst_img)
                total_copied[split] += 1

            lbl = find_label_for_image(img, src_lbl_dir)
            if lbl and lbl.exists():
                dst_lbl = dst_lbl_dir / f"{img.stem}.txt"
                if dst_lbl.exists() and not force:
                    if dst_lbl.stat().st_size == lbl.stat().st_size:
                        pass
                    else:
                        shutil.copy2(lbl, dst_lbl)
                        total_labels[split] += 1
                else:
                    shutil.copy2(lbl, dst_lbl)
                    total_labels[split] += 1
            else:
                total_bg[split] += 1

        # también contar huérfanos (labels sin imagen) solo informativo
        orphan_labels = []
        if src_lbl_dir.exists():
            for txt in src_lbl_dir.rglob("*.txt"):
                stem = txt.stem
                # si no existe imagen con ese stem en este split, es huérfano
                if not any((src_img_dir.rglob(f"{stem}.*"))):
                    # check exact: busca imagen con ese stem
                    found = list(src_img_dir.rglob(f"{stem}.jpg")) + list(src_img_dir.rglob(f"{stem}.png")) + list(src_img_dir.rglob(f"{stem}.jpeg"))
                    if not found:
                        orphan_labels.append(txt)
        if orphan_labels:
            print(f"[{split}] labels huérfanos sin imagen (se omiten): {len(orphan_labels)} ej: {[p.name for p in orphan_labels[:3]]}")

        print(f"[{split}] copiadas imágenes nuevas: {total_copied[split]}  labels copiados: {total_labels[split]}  fondo sin label: {total_bg[split]}")

    # verificación con glob plano (lo que verá ultralytics)
    for split in ("train", "val"):
        flat_imgs = list((DST_BASE / split / "images").glob("*.*"))
        flat_imgs = [p for p in flat_imgs if p.suffix.lower() in IMAGE_EXTS]
        flat_lbls = list((DST_BASE / split / "labels").glob("*.txt"))
        print(f"[verify] {split} plano: images={len(flat_imgs)} labels={len(flat_lbls)}")

    # genera data.yaml
    generate_yaml()
    print(f"[prepare_hornet] data.yaml generado en {DATA_YAML}")
    return 0


def generate_yaml():
    DST_BASE.mkdir(parents=True, exist_ok=True)
    # path absoluto con forward slashes para YOLO
    abs_path = DST_BASE.resolve().as_posix()
    # también acepta relativo, pero usamos absoluto para robustez
    content = (
        f"path: {abs_path}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"nc: {len(NAMES)}\n"
        f"names: [{', '.join(NAMES)}]\n"
    )
    DATA_YAML.write_text(content, encoding="utf-8")
    # verificación con glob
    import glob as _glob
    for split in ("train", "val"):
        pattern = str(DST_BASE / split / "images" / "*.*")
        found = _glob.glob(pattern)
        # filtra imágenes
        found = [f for f in found if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
        print(f"[yaml verify] {split}/images glob -> {len(found)} ficheros (espera >0)")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Prepara Hornet3000+ plano para YOLO")
    ap.add_argument("--check", action="store_true", help="solo lista conteos, no copia")
    ap.add_argument("--force", action="store_true", help="fuerza recopia aunque exista")
    args = ap.parse_args(argv)
    if args.check:
        stats = count_real(SRC_BASE)
        for (split, kind), v in stats.items():
            print(f"{split}/{kind}: subfolders={v['subfolders']} flat={v['flat']} recursive={v['recursive']}")
        # glob check
        for split in ("train", "val"):
            for kind in ("images", "labels"):
                p = DST_BASE / split / kind
                if p.exists():
                    n = len(list(p.glob("*.*")))
                    print(f"dst {split}/{kind}: {n}")
                else:
                    print(f"dst {split}/{kind}: no existe")
        if DATA_YAML.exists():
            print(f"data.yaml existe:\n{DATA_YAML.read_text(encoding='utf-8')}")
        else:
            print("data.yaml no existe")
        return 0
    return prepare(force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
