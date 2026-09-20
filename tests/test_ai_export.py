"""RED 3.1a — PR3 export ONNX opset12 CPU 320 (falla sin export/export_onnx.py)."""
import numpy as np


def test_export_opset12_simplify_and_shape_check(tmp_path):
    from jetson.ai.export import export_onnx

    ckpt = tmp_path / "ckpt2.pt"
    ckpt.write_text("primary: yolov8s-seg\ninit_weights: COCO\n", encoding="utf-8")
    out = tmp_path / "velutina.onnx"
    artifact = export_onnx.export_ckpt(ckpt=str(ckpt), out=str(out))
    assert out.exists(), "export must write .onnx artifact"
    meta = export_onnx.read_meta(str(out))
    assert meta["opset"] == 12, f"opset must be 12, got {meta.get('opset')}"
    assert meta["simplified"] is True
    assert meta["input_shape"] == [1, 3, 320, 320]
    # check CPU 320x320 sin GPU/onnxruntime pesado
    ok = export_onnx.check(str(out), size=320)
    assert ok is True


def test_export_trtexec_is_doc_only_no_engine_no_int8(tmp_path):
    from jetson.ai.export import export_onnx

    cmd = export_onnx.trtexec_command("velutina.onnx", "velutina_fp16.engine")
    assert "--fp16" in cmd and "--onnx=" in cmd, f"trtexec fp16 doc missing: {cmd}"
    assert "velutina.onnx" in cmd
    assert "--int8" not in cmd.lower(), "must not request INT8 quantization flag"
    # nunca ejecuta subprocess ni crea .engine
    assert not (tmp_path / "velutina_fp16.engine").exists()
    assert export_onnx.uses_subprocess() is False
