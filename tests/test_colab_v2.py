"""Valida el cuaderno Colab v2 y la extension --umbral de eval_yolo. Rapido (<15s, sin red)."""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB2 = REPO / "notebooks" / "train_yolo_colab_v2.ipynb"
NB1 = REPO / "notebooks" / "train_yolo_colab.ipynb"
EVAL = REPO / "sim" / "dataset" / "eval_yolo.py"

REQUIRED_STRINGS = ["yolov8s-cls", "operating_point", "ATTRIBUTION", "kaggle.json", "Eristalis"]


def _nb_text(nb):
    return "\n".join(
        "".join(c.get("source", []))
        for c in nb.get("cells", [])
    )


def test_v2_exists_and_valid_json():
    assert NB2.is_file(), f"falta {NB2}"
    data = json.loads(NB2.read_text(encoding="utf-8"))
    assert data.get("nbformat") == 4, "nbformat debe ser 4"
    assert "cells" in data and len(data["cells"]) > 0


def test_v2_has_enough_code_cells():
    data = json.loads(NB2.read_text(encoding="utf-8"))
    code = [c for c in data["cells"] if c.get("cell_type") == "code"]
    assert len(code) >= 7, f"se esperaban >=7 celdas de codigo, hay {len(code)}"


def test_v2_contains_key_strings():
    data = json.loads(NB2.read_text(encoding="utf-8"))
    text = _nb_text(data)
    for s in REQUIRED_STRINGS:
        assert s in text, f"falta string clave: {s}"


def test_v1_untouched():
    assert NB1.is_file(), f"v1 debe seguir intacto en {NB1}"
    data = json.loads(NB1.read_text(encoding="utf-8"))
    text = _nb_text(data)
    assert "yolov8n-cls" in text, "v1 debe seguir usando yolov8n-cls"
    assert "operating_point" not in text, "v1 no debe contener calibracion v2"


def test_eval_help_works_and_has_umbral():
    assert EVAL.is_file(), f"falta {EVAL}"
    r = subprocess.run(
        [sys.executable, str(EVAL), "--help"],
        capture_output=True, text=True, timeout=60, cwd=str(REPO),
    )
    assert r.returncode == 0, f"--help fallo: {r.stderr[:500]}"
    assert "--umbral" in r.stdout, "eval_yolo.py debe aceptar --umbral"
    assert "--weights" in r.stdout, "eval_yolo.py debe aceptar --weights"
