"""Valida cuaderno Kaggle manos libres y guia KAGGLE.md. Rapido (<15s, sin red)."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB = REPO / "notebooks" / "train_yolo_detect_kaggle_handsfree.ipynb"
DOC = REPO / "docs" / "KAGGLE.md"

REQUIRED_STRINGS = ["ultralytics", "/kaggle/input", "/kaggle/working", "best.pt", "resultados_det"]
# hornet3000 o vespa-velutina debe aparecer
ALT_STRINGS = ["hornet3000", "vespa-velutina"]


def _nb_text(nb):
    return "\n".join("".join(c.get("source", [])) for c in nb.get("cells", []))


def test_notebook_exists_and_valid_json():
    assert NB.is_file(), f"falta {NB}"
    data = json.loads(NB.read_text(encoding="utf-8"))
    assert data.get("nbformat") == 4, "nbformat debe ser 4"
    assert "cells" in data and len(data["cells"]) > 0


def test_notebook_has_enough_code_cells():
    data = json.loads(NB.read_text(encoding="utf-8"))
    code = [c for c in data["cells"] if c.get("cell_type") == "code"]
    assert len(code) >= 5, f"se esperaban >=5 celdas de codigo, hay {len(code)}"


def test_notebook_contains_key_strings():
    data = json.loads(NB.read_text(encoding="utf-8"))
    text = _nb_text(data).lower()
    for s in REQUIRED_STRINGS:
        assert s.lower() in text, f"falta string clave en ipynb: {s}"
    assert any(a.lower() in text for a in ALT_STRINGS), f"falta hornet3000 o vespa-velutina en ipynb"


def test_kaggle_doc_exists_and_mentions_gpu_and_apagar():
    assert DOC.is_file(), f"falta {DOC}"
    text = DOC.read_text(encoding="utf-8")
    low = text.lower()
    assert "gpu" in low, "docs/KAGGLE.md debe mencionar GPU"
    # apagar PC (con espacio, case-insensitive)
    assert "apagar" in low and "pc" in low, "docs/KAGGLE.md debe mencionar apagar PC"
