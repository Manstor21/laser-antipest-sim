"""Valida el cuaderno Colab y su guia. Debe ir en <10s (sin red)."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB = REPO / "notebooks" / "train_yolo_colab.ipynb"
DOC = REPO / "docs" / "COLAB.md"

REQUIRED_STRINGS = ["ultralytics", "gbif", "yolov8n-cls", "onnx", "ATTRIBUTION"]


def _nb_text(nb):
    return "\n".join(
        "".join(c.get("source", []))
        for c in nb.get("cells", [])
    )


def test_notebook_exists_and_valid_json():
    assert NB.is_file(), f"falta {NB}"
    data = json.loads(NB.read_text(encoding="utf-8"))
    assert data.get("nbformat") == 4, "nbformat debe ser 4"
    assert "cells" in data and len(data["cells"]) > 0


def test_notebook_has_enough_code_cells():
    data = json.loads(NB.read_text(encoding="utf-8"))
    code = [c for c in data["cells"] if c.get("cell_type") == "code"]
    assert len(code) >= 6, f"se esperaban >=6 celdas de codigo, hay {len(code)}"


def test_notebook_contains_key_strings():
    data = json.loads(NB.read_text(encoding="utf-8"))
    text = _nb_text(data)
    for s in REQUIRED_STRINGS:
        assert s in text, f"falta string clave: {s}"


def test_colab_doc_exists_and_mentions_free_gpu():
    assert DOC.is_file(), f"falta {DOC}"
    text = DOC.read_text(encoding="utf-8")
    assert "GPU gratis" in text, "docs/COLAB.md debe mencionar GPU gratis"
