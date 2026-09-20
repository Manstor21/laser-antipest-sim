"""C1 fix verify #149 — Training starts from COCO (v8s primario + v8n baseline).

Cubre jetson/ai/train.py::train_round sin covering pytest previo.
TDD: RED = sin train.py fallaba con ImportError; GREEN = ckpt+log verifican COCO init.
"""
from pathlib import Path

import yaml


def _write_cfg(tmp_path: Path, name: str = "round1.yaml") -> Path:
    cfg = tmp_path / name
    cfg.write_text(
        yaml.safe_dump({"ohem_round": 1, "train": "sim/dataset"}),
        encoding="utf-8",
    )
    return cfg


def test_train_round_ckpt_contains_coco_init_and_v8s_primary(tmp_path):
    from jetson.ai import train

    cfg = _write_cfg(tmp_path)
    out = tmp_path / "ckpt1.pt"
    result = train.train_round(str(cfg), str(out))
    assert Path(result).exists(), "train_round() must write ckpt"
    payload = Path(result).read_text(encoding="utf-8")
    # init COCO explícito (literal spec, no solo constante)
    assert "COCO" in payload, f"ckpt must record COCO init: {payload!r}"
    assert "init_weights: COCO" in payload, f"ckpt must carry init_weights COCO: {payload!r}"
    # primario v8s-seg explícito
    assert "yolov8s-seg" in payload, f"ckpt must record primary v8s-seg: {payload!r}"
    assert "primary: yolov8s-seg" in payload, f"ckpt must mark primary: {payload!r}"
    # constantes del módulo alineadas con spec
    assert train.PRIMARY == "yolov8s-seg"
    assert train.INIT == "COCO"


def test_train_round_log_contains_v8s_and_v8n_baseline(tmp_path):
    from jetson.ai import train

    cfg = _write_cfg(tmp_path)
    out = tmp_path / "ckpt1.pt"
    result = train.train_round(str(cfg), str(out))
    log_p = Path(result).parent / (Path(result).stem + ".log")
    assert log_p.exists(), "train_round() must write sidecar .log"
    log = log_p.read_text(encoding="utf-8")
    # log v8s primario + v8n baseline + COCO
    assert "yolov8s-seg" in log, f"log must mention primary v8s: {log!r}"
    assert "yolov8n-seg" in log, f"log must mention baseline v8n: {log!r}"
    assert "COCO" in log, f"log must mention COCO init: {log!r}"
    assert train.BASELINE == "yolov8n-seg"
