"""Tests del replay GBIF sin bboxes (crop central 320, <30s)."""

import numpy as np
import pytest

from sim.dataset import replay_gbif as rg


def _make_fixture(tmp_path):
    """Estructura dirs con 2 PNG: velutina brillante + abeja oscura."""
    import cv2

    d_vel = tmp_path / "gbif_velutina"
    d_abe = tmp_path / "gbif_abeja"
    d_vel.mkdir(parents=True, exist_ok=True)
    d_abe.mkdir(parents=True, exist_ok=True)

    bright = np.full((480, 640, 3), 200, dtype=np.uint8)
    bright[140:340, 220:420] = 255
    cv2.imwrite(str(d_vel / "a.jpg"), bright)

    dark = np.full((480, 640, 3), 20, dtype=np.uint8)
    cv2.imwrite(str(d_abe / "b.jpg"), dark)
    return d_vel, d_abe


def test_crop_central_sale_320x320x3(tmp_path):
    _make_fixture(tmp_path)
    img = rg.load_image_rgb(tmp_path / "gbif_velutina" / "a.jpg")
    crop = rg.make_center_crop_320(img)
    assert crop.shape == (320, 320, 3)
    assert crop.dtype == np.uint8
    # Imagen pequeña: reescala lado mayor a 320 y rellena.
    tiny = np.full((50, 100, 3), 128, dtype=np.uint8)
    crop2 = rg.make_center_crop_320(tiny)
    assert crop2.shape == (320, 320, 3)


def test_informe_contiene_recall(tmp_path, capsys):
    d_vel, d_abe = _make_fixture(tmp_path)
    rc = rg.main(["--dirs", str(d_vel), str(d_abe), "--limit", "10"])
    assert rc == 0
    out = capsys.readouterr().out.lower()
    assert "recall" in out
    assert "velutina" in out


def test_sin_dirs_sale_exit2_con_ayuda(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        rg.main(["--dirs", str(tmp_path / "noexiste"), "--limit", "10"])
    assert exc.value.code == 2
    text = (capsys.readouterr().out + capsys.readouterr().err).lower()
    assert ("gbif" in text or "descarga" in text or "download" in text
            or "fetch" in text or "fotos" in text)
