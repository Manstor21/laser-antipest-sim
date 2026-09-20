"""Tests del detector local YOLO (sin entreno, <60s)."""
from __future__ import annotations

import pytest
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = REPO_ROOT / "data" / "det" / "data.yaml"
BEST = REPO_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"
DET_PT = REPO_ROOT / "jetson" / "ai" / "yolov8n_det.pt"


def test_data_yaml_existe_y_apunta_a_dirs_con_imagenes():
    assert DATA_YAML.exists(), f"falta {DATA_YAML}"
    text = DATA_YAML.read_text(encoding="utf-8")
    assert "vespa_velutina" in text
    assert "vespa_crabro" in text
    assert "vespula_vulgaris" in text
    # parse simple
    import yaml
    data = yaml.safe_load(text)
    assert "path" in data or "train" in data
    # resolve paths
    # data.yaml may have absolute path
    base = Path(data["path"]) if "path" in data else REPO_ROOT / "data" / "det"
    if not base.is_absolute():
        base = (REPO_ROOT / base).resolve()
    train_dir = base / data.get("train", "train/images")
    val_dir = base / data.get("val", "val/images")
    assert train_dir.exists(), f"train dir no existe {train_dir}"
    assert val_dir.exists(), f"val dir no existe {val_dir}"
    train_imgs = list(train_dir.glob("*.*"))
    val_imgs = list(val_dir.glob("*.*"))
    # filtra imagenes
    train_imgs = [p for p in train_imgs if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    val_imgs = [p for p in val_imgs if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    assert len(train_imgs) >= 3000, f"train imgs muy pocas {len(train_imgs)}"
    assert len(val_imgs) >= 200, f"val imgs muy pocas {len(val_imgs)}"
    # check flat: no subcarpetas con imagenes
    assert not any((train_dir / d).is_dir() for d in ["Vespa_velutina", "Vespa_crabro"]), "debe estar aplanado"


def test_best_pt_cargable_si_existe():
    if not BEST.exists():
        pytest.skip(f"sin best.pt ({BEST}), skip (aún no entrenado)")
    from ultralytics import YOLO
    assert BEST.stat().st_size > 1_000_000, "best.pt demasiado pequeño"
    m = YOLO(str(BEST))
    assert hasattr(m, "names")
    assert "vespa_velutina" in {v for v in m.names.values()} or "Vespa_velutina" in {v for v in m.names.values()} or 0 in m.names


def test_det_backend_load_predict_sintetico():
    from jetson.ai import det_backend as db
    # si no existe det pt, debe dar error claro en load, pero test debe cubrir ambos casos
    if not DET_PT.exists():
        pytest.skip(f"sin det pt {DET_PT}, skip")
    # reset cache por si otro test lo tocó
    db.reset_cache()
    m = db.load()
    assert "model" in m
    assert "names" in m
    # fixture sintética 320x320 o 640x480: detector debe aceptarla sin error (puede devolver vacía)
    img = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
    boxes = db.detect(img, conf=0.99)  # umbral alto para no forzar FP
    assert isinstance(boxes, list)
    # también con numpy 320
    img2 = np.zeros((320, 320, 3), dtype=np.uint8)
    img2[100:200, 100:200] = 255
    boxes2 = db.detect(img2, conf=0.25)
    assert isinstance(boxes2, list)
    # cada box si existe debe tener claves
    for b in boxes2:
        assert "xyxy" in b and "conf" in b and "cls" in b and "name" in b
        assert isinstance(b["xyxy"], list) and len(b["xyxy"]) == 4
        assert 0 <= b["conf"] <= 1
    db.reset_cache()


def test_det_backend_error_claro_sin_peso(monkeypatch, tmp_path):
    from jetson.ai import det_backend as db
    monkeypatch.setattr(db, "MODEL_PATH", tmp_path / "no_existe.pt")
    db.reset_cache()
    try:
        with pytest.raises(FileNotFoundError):
            db.load()
        # detect también debe fallar con FileNotFound
        img = np.zeros((320, 320, 3), dtype=np.uint8)
        with pytest.raises(FileNotFoundError):
            db.detect(img)
    finally:
        db.reset_cache()


def test_resultados_y_box_json_existen():
    txt = REPO_ROOT / "resultados_det_local.txt"
    js = REPO_ROOT / "jetson" / "ai" / "box_thresholds_det.json"
    assert txt.exists(), f"falta {txt}"
    assert js.exists(), f"falta {js}"
    assert "mAP" in txt.read_text(encoding="utf-8")
    import json
    data = json.loads(js.read_text(encoding="utf-8"))
    assert "umbral" in data and "mAP50" in data
