"""Tests del replay sim-to-real con dataset YOLO sintético (<30s)."""

import numpy as np
import pytest

from sim.dataset import replay_real as rr


def _make_fixture(tmp_path):
    """2 PNG 640x480 + labels YOLO + data.yaml con [vespa_velutina, apis]."""
    import cv2

    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    # Clara con rectángulo brillante = fake velutina (mucho foreground).
    bright = np.full((480, 640, 3), 200, dtype=np.uint8)
    bright[140:340, 220:420] = 255
    cv2.imwrite(str(img_dir / "velu.png"), bright)
    (lbl_dir / "velu.txt").write_text("0 0.5 0.5 0.3 0.4\n", encoding="utf-8")

    # Oscura = fake fondo (poco foreground).
    dark = np.full((480, 640, 3), 20, dtype=np.uint8)
    cv2.imwrite(str(img_dir / "fondo.png"), dark)
    (lbl_dir / "fondo.txt").write_text("1 0.5 0.5 0.3 0.4\n", encoding="utf-8")

    (tmp_path / "data.yaml").write_text(
        "path: .\ntrain: images\nval: images\n"
        "names: [vespa_velutina, apis_mellifera]\n",
        encoding="utf-8",
    )
    return tmp_path


def test_parseo_encuentra_dos_cajas(tmp_path):
    d = _make_fixture(tmp_path)
    boxes = rr.collect_boxes(d, "val", limit=200)
    assert len(boxes) == 2


def test_crop_sale_320x320x3(tmp_path):
    d = _make_fixture(tmp_path)
    boxes = rr.collect_boxes(d, "val", limit=200)
    img_path, _, cx, cy, w, h = boxes[0]
    img = rr.load_image_rgb(img_path)
    crop = rr.make_square_crop_320(img, cx, cy, w, h)
    assert crop.shape == (320, 320, 3)
    assert crop.dtype == np.uint8


def test_informe_contiene_recall_y_dos_procesadas(tmp_path, capsys):
    d = _make_fixture(tmp_path)
    rc = rr.main(["--data", str(d), "--limit", "200", "--split", "val"])
    assert rc == 0
    out = capsys.readouterr().out.lower()
    assert "recall" in out
    assert "2 procesadas" in out


def test_sin_data_sale_exit2_con_ayuda(capsys):
    with pytest.raises(SystemExit) as exc:
        rr.main([])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    text = (captured.out + captured.err).lower()
    assert ("kaggle" in text or "roboflow" in text or "descarga" in text
            or "download" in text)
