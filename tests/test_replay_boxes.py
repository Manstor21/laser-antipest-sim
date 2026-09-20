"""Tests de sim/dataset/replay_boxes.py con fixture temporal (<30s, sin pesos)."""

from __future__ import annotations

import numpy as np

from sim.dataset import replay_boxes as rb


def _make_fixture(tmp_path):
    """1 PNG 640x480 + 1 label YOLO con 2 cajas + config mínima (0/1/2)."""
    import cv2

    img_dir = tmp_path / "val" / "images" / "Vespa_velutina"
    lbl_dir = tmp_path / "val" / "labels" / "Vespa_velutina"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    img = np.full((480, 640, 3), 128, dtype=np.uint8)
    img[140:340, 220:420] = 200
    cv2.imwrite(str(img_dir / "a.png"), img)
    (lbl_dir / "a.txt").write_text(
        "0 0.5 0.5 0.3 0.4\n1 0.2 0.2 0.1 0.1\n", encoding="utf-8")

    (tmp_path / "config.yaml").write_text(
        "names:\n  0: Vespa_velutina\n  1: Vespa_crabro\n"
        "  2: Vespula_vulgaris\n",
        encoding="utf-8",
    )
    return tmp_path


def test_parseo_encuentra_dos_cajas(tmp_path):
    d = _make_fixture(tmp_path)
    boxes = rb.collect_boxes(d, "val", limit=0)
    assert len(boxes) == 2
    assert boxes[0][1] == 0  # velutina
    assert boxes[1][1] == 1  # crabro


def test_crop_sale_320x320x3(tmp_path):
    d = _make_fixture(tmp_path)
    boxes = rb.collect_boxes(d, "val", limit=0)
    img_path, _, cx, cy, w, h = boxes[0]
    img = rb.load_image_rgb(img_path)
    crop = rb.make_square_crop_320(img, cx, cy, w, h)
    assert crop.shape == (320, 320, 3)
    assert crop.dtype == np.uint8


def test_umbrales_por_defecto():
    assert rb.default_threshold_for("jetson/ai/yolov8n_velutina.pt") == 0.9994
    assert rb.default_threshold_for("jetson/ai/yolov8s_velutina_v2.pt") == 0.9958


def test_informe_contiene_recall_con_stub(tmp_path, capsys):
    d = _make_fixture(tmp_path)
    names = rb.load_class_names(d)
    items = rb.collect_boxes(d, "val", limit=0)
    # Stub: la velutina (cls 0) promociona, el crabro (cls 1) no.
    pvs = iter([0.9999, 0.10])
    stats_v1 = rb.run_replay(items, names, lambda _c: next(pvs), 0.9994)
    pvs2 = iter([0.9999, 0.10])
    stats_v2 = rb.run_replay(items, names, lambda _c: next(pvs2), 0.9958)
    text = rb.build_report_text(
        {"v1": stats_v1, "v2": stats_v2},
        {"velutina": 1, "crabro": 1, "vulgaris": 0}, "val", 1)
    low = text.lower()
    assert "recall" in low
    assert "diagn" in low  # etiqueta DIAGNÓSTICO
    assert "veredicto" in low
    assert stats_v1["recall"] == 1.0
    assert stats_v1.get("fp_crabro", 0) == 0
    print(text)
    out = capsys.readouterr().out.lower()
    assert "recall" in out


def test_sweep_encuentra_mejor_fp0_sintetico(tmp_path):
    """Sweep sintético: verifica mayor recall con FP=0 y desempate a menor umbral."""
    from pathlib import Path
    # 3 velutina, 2 crabro, 1 vulgaris con Pv controlados
    # velutina Pv altas, negativas bajas tras 0.95
    items = [
        (Path("img_a.jpg"), 0, 0.5, 0.5, 0.1, 0.1),  # velutina 0.99
        (Path("img_b.jpg"), 0, 0.5, 0.5, 0.1, 0.1),  # velutina 0.96
        (Path("img_c.jpg"), 0, 0.5, 0.5, 0.1, 0.1),  # velutina 0.92
        (Path("img_d.jpg"), 1, 0.5, 0.5, 0.1, 0.1),  # crabro 0.94
        (Path("img_e.jpg"), 1, 0.5, 0.5, 0.1, 0.1),  # crabro 0.85
        (Path("img_f.jpg"), 2, 0.5, 0.5, 0.1, 0.1),  # vulgaris 0.91
    ]
    names = {0: "Vespa_velutina", 1: "Vespa_crabro", 2: "Vespula_vulgaris"}
    pvs = [0.99, 0.96, 0.92, 0.94, 0.85, 0.91]
    rows = rb.compute_sweep_rows(items, names, pvs)
    assert len(rows) == len(rb.SWEEP_THRESHOLDS)
    # En 0.95 -> 2/3 recall, FP=0; en 0.93 -> FP>0, en 0.97 -> 1/3 recall
    best = rb.pick_best_fp0(rows)
    assert best is not None
    assert best["fp"] == 0
    assert abs(best["umbral"] - 0.95) < 1e-6
    assert abs(best["recall"] - (2/3)) < 1e-6
    # Empate: dos umbrales con mismo recall FP=0 -> gana menor umbral
    items2 = [
        (Path("img_a.jpg"), 0, 0.5, 0.5, 0.1, 0.1),  # velutina 0.99
        (Path("img_b.jpg"), 1, 0.5, 0.5, 0.1, 0.1),  # crabro 0.94
    ]
    pvs2 = [0.99, 0.94]
    # Umbrales sintéticos con mismo recall
    thr_pair = [0.95, 0.97]
    rows2 = rb.compute_sweep_rows(items2, names, pvs2, thresholds=thr_pair)
    # Ambos FP=0 y recall=1.0 -> debe elegir 0.95
    best2 = rb.pick_best_fp0(rows2)
    assert best2 is not None
    assert abs(best2["umbral"] - 0.95) < 1e-6
    # Tabla marca MEJOR
    tbl = rb.format_sweep_table("v1", rows, best)
    assert "MEJOR FP=0" in tbl
    assert "0.9500" in tbl


def test_sweep_guarda_json_con_claves_por_modelo(tmp_path):
    """Json box_thresholds con claves v1/v2/n_cajas/fecha y subclaves por modelo."""
    import json
    from pathlib import Path
    items = [
        (Path("a.jpg"), 0, 0.5, 0.5, 0.1, 0.1),
        (Path("b.jpg"), 1, 0.5, 0.5, 0.1, 0.1),
    ]
    names = {0: "Vespa_velutina", 1: "Vespa_crabro", 2: "Vespula_vulgaris"}
    # v1: FP=0 en 0.95 (recall alto), v2: simulamos otro perfil
    pvs_v1 = [0.99, 0.94]  # best 0.95 recall 1.0 FP0
    pvs_v2 = [0.96, 0.80]  # best 0.85 recall 1.0 FP0 -> el menor FP0 con max recall es 0.5 en este caso
    rows_v1 = rb.compute_sweep_rows(items, names, pvs_v1)
    rows_v2 = rb.compute_sweep_rows(items, names, pvs_v2)
    best_v1 = rb.pick_best_fp0(rows_v1)
    best_v2 = rb.pick_best_fp0(rows_v2)
    assert best_v1 is not None and best_v2 is not None
    out = tmp_path / "box_thresholds.json"
    rb.save_box_thresholds({"v1": best_v1, "v2": best_v2}, n_cajas=len(items), out_path=out)
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "v1" in data and "v2" in data
    assert "n_cajas" in data and "fecha" in data
    for k in ("v1", "v2"):
        assert "umbral" in data[k]
        assert "recall" in data[k]
        assert "fp" in data[k]
        assert "n" in data[k]
    assert data["n_cajas"] == len(items)
    assert isinstance(data["fecha"], str) and len(data["fecha"]) > 5
    assert data["v1"]["fp"] == 0 and data["v2"]["fp"] == 0
    # Verifica que el json real usa la misma función y claves
    assert float(data["v1"]["umbral"]) == best_v1["umbral"]
    assert abs(float(data["v1"]["recall"]) - best_v1["recall"]) < 1e-6
