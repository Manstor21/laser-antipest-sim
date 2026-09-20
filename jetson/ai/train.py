"""PR2 core: entrenamiento CPU sim-only 2 rondas (0EUR, sin GPU).

`train_round(cfg)` cfg round1.yaml -> ckpt1.pt, round2.yaml -> ckpt2.pt.
Init COCO, primario v8s + baseline v8n (solo log, sin descarga).
CLI: `python jetson/ai/train.py --round 1|2 [--data manifest] [--dry-run]`.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import yaml

PRIMARY = "yolov8s-seg"
BASELINE = "yolov8n-seg"
INIT = "COCO"


def _ckpt_name_for_cfg(cfg_path: Path) -> str:
    name = cfg_path.stem.lower()
    if "round2" in name or "round-2" in name or name.endswith("2"):
        return "ckpt2.pt"
    return "ckpt1.pt"


def train_round(cfg: str, out: str | None = None) -> str:
    """Entrena (simulado CPU) desde cfg -> ckpt. Retorna ruta del ckpt."""
    cfg_p = Path(cfg)
    if not cfg_p.exists():
        raise FileNotFoundError(f"missing train cfg {cfg_p}")
    with open(cfg_p, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    out_p = Path(out) if out else (cfg_p.parent / _ckpt_name_for_cfg(cfg_p))
    out_p.parent.mkdir(parents=True, exist_ok=True)
    # Checkpoint simulado: payload yaml con sufijo .pt (0€, sin torch).
    payload = (
        f"# sim-ckpt {PRIMARY} init={INIT} cpu-only\n"
        f"primary: {PRIMARY}\n"
        f"baseline: {BASELINE}\n"
        f"init_weights: {INIT}\n"
        f"device: cpu\n"
        f"cfg: {cfg_p.name}\n"
        f"ohem_round: {data.get('ohem_round', 1)}\n"
        f"trained_at: {datetime.now(timezone.utc).isoformat()}\n"
    )
    out_p.write_text(payload, encoding="utf-8")
    log_p = out_p.parent / (out_p.stem + ".log")
    log_p.write_text(
        f"train {PRIMARY} from {INIT} on CPU (primary) + {BASELINE} baseline\ncfg={cfg_p}\n",
        encoding="utf-8",
    )
    return str(out_p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train YOLOv8s-seg CPU sim-only (OHEM 2 rounds)")
    ap.add_argument("--round", required=True, choices=["1", "2"], help="OHEM round")
    ap.add_argument(
        "--data",
        default=str(Path(__file__).resolve().parents[2] / "sim" / "dataset" / "dataset_manifest.yaml"),
        help="train manifest (round1) or round2.yaml (round2)",
    )
    ap.add_argument("--out", default=None, help="ckpt output path")
    ap.add_argument("--dry-run", action="store_true", help="validate without writing ckpt")
    args = ap.parse_args(argv)

    cfg_p = Path(args.data)
    print(f"train {PRIMARY} (baseline {BASELINE}) init={INIT} device=cpu round={args.round} cfg={cfg_p}")
    if args.dry_run:
        if not cfg_p.exists():
            print(f"dry-run: missing cfg {cfg_p}")
            return 2
        print("dry-run: ok (no ckpt written)")
        return 0
    ckpt = train_round(str(cfg_p), args.out)
    print(f"wrote {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
