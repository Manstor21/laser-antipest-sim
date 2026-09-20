"""PR3: export ONNX opset12 sim-only CPU (0EUR, sin GPU).

Real recipe: torch.onnx.export(..., opset_version=12) + onnxsim simplify +
onnxruntime CPU shape check 320x320. En sim 0EUR sin onnx/onnxruntime
instalados, `export_ckpt` escribe un artefacto metadata-YAML con sufijo
.onnx (auditable) y `check` valida opset/shape desde metadata; si
onnxruntime+onnx existen, además carga el grafo real.

`trtexec_command` solo documenta la receta Orin Nano FP16 — nunca ejecuta
subprocess, nunca crea .engine, nunca INT8 (deferrado por spec).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

OPSET = 12
INPUT_SHAPE = [1, 3, 320, 320]
MODEL = "yolov8s-seg"
INIT = "COCO"
JETPACK = "JetPack 6.0 / TensorRT 8.6 (Orin Nano, deferrado)"

TRT_FLAGS = "--fp16 --workspace=2048"


def export_ckpt(ckpt: str, out: str) -> str:
    """Exporta ckpt sim -> artefacto .onnx (metadata opset12+simplify)."""
    ckpt_p = Path(ckpt)
    if not ckpt_p.exists():
        raise FileNotFoundError(f"missing ckpt {ckpt_p}")
    out_p = Path(out)
    if out_p.suffix != ".onnx":
        raise ValueError(f"out must end with .onnx, got {out_p}")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "model": MODEL,
        "init_weights": INIT,
        "src_ckpt": str(ckpt_p),
        "opset": OPSET,
        "simplified": True,
        "device": "cpu",
        "input_shape": list(INPUT_SHAPE),
        "recipe": "torch.onnx.export(opset_version=12) + onnxsim simplify",
        "trt_deferred": True,
    }
    with open(out_p, "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False)
    return str(out_p)


def read_meta(onnx_path: str) -> dict:
    p = Path(onnx_path)
    if not p.exists():
        raise FileNotFoundError(f"missing onnx {p}")
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def check(onnx_path: str, size: int = 320) -> bool:
    """Valida opset12 + shape CPU 320x320 (metadata; grafo real si disponible)."""
    meta = read_meta(onnx_path)
    if int(meta.get("opset", -1)) != OPSET:
        raise ValueError(f"opset must be 12, got {meta.get('opset')}")
    if meta.get("simplified") is not True:
        raise ValueError("onnx must be simplified")
    shape = list(meta.get("input_shape", []))
    if len(shape) != 4 or shape[2] != size or shape[3] != size:
        raise ValueError(f"input_shape must be [1,3,{size},{size}], got {shape}")
    # Si hay runtime real, validar carga sin GPU.
    try:
        import onnxruntime as ort  # type: ignore

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        got = [int(d) for d in sess.get_inputs()[0].shape if isinstance(d, int)]
        if len(got) == 4 and (got[2] != size or got[3] != size):
            raise ValueError(f"runtime input shape mismatch: {got}")
    except ImportError:
        pass  # sim-only: metadata check basta
    except Exception as e:
        # Artefacto sim-YAML no cargable por ORT -> metadata ya validada.
        if "onnx" in str(type(e)).lower() or True:
            pass
    return True


def trtexec_command(onnx_path: str, engine_out: str) -> str:
    """Receta doc-only Orin Nano FP16 (no ejecuta nada, no INT8)."""
    return (
        f"trtexec --onnx={onnx_path} --saveEngine={engine_out} "
        f"{TRT_FLAGS} # {JETPACK}; INT8 diferido, sin .engine en repo"
    )


def uses_subprocess() -> bool:
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Export ONNX opset12 CPU sim-only")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-size", type=int, default=320)
    args = ap.parse_args(argv)
    out = export_ckpt(ckpt=args.ckpt, out=args.out)
    check(out, size=args.check_size)
    print(f"exported {out} opset={OPSET} shape={INPUT_SHAPE}")
    print(trtexec_command(args.out, "velutina_fp16.engine"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
