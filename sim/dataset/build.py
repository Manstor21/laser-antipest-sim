"""PR1 foundation: curated-combined + Gazebo DR 320 manifest builder (sim-only, 0EUR)."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml

IMAGE_SIZE = 320
DR_VARIANTS = ("rain", "glare", "occlusion", "clean")

SOURCES = [
    {"name": "vespai-protocol", "license": "CC-BY-4.0", "n": 2100},
    {"name": "roboflow-filtered", "license": "CC-BY-4.0", "n": 2500},
    {"name": "bee-vs-wasp-neg", "license": "CC-BY-4.0", "n": 800},
    {"name": "gazebo-synth-320", "license": "proprietary-sim", "n": 3000},
]

ATTRIBUTION = [
    "VespAI-protocol (CC-BY-4.0) — courtesy of original authors, see manifest sources",
    "Roboflow filtered subset (CC-BY-4.0) — courtesy of original authors",
    "Bee-vs-Wasp negatives (CC-BY-4.0) — courtesy of original authors",
]

SYNTH_NAME = "gazebo-synth-320"


def dedup_sources(sources=None):
    """Dedup combo by source name, preserving first occurrence."""
    srcs = list(SOURCES if sources is None else sources)
    seen = set()
    out = []
    for s in srcs:
        if s["name"] not in seen:
            seen.add(s["name"])
            out.append(dict(s))
    return out


def compute_synthetic_ratio(sources=None) -> float:
    srcs = dedup_sources(sources)
    total = sum(s["n"] for s in srcs)
    synth = next(s["n"] for s in srcs if s["name"] == SYNTH_NAME)
    return synth / total if total else 0.0


def build_manifest_dict(sources=None) -> dict:
    srcs = dedup_sources(sources)
    return {
        "image_size": [IMAGE_SIZE, IMAGE_SIZE],
        "dr_variants": list(DR_VARIANTS),
        "sources": srcs,
        "synthetic_ratio": compute_synthetic_ratio(srcs),
        "attribution": list(ATTRIBUTION),
    }


def generate_manifest(out: str | Path) -> str:
    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        yaml.safe_dump(build_manifest_dict(), f, sort_keys=False)
    return str(out_p)


def sample_batch(n: int = 12, seed: int = 0) -> list[dict]:
    """Sample a synthetic DR batch: 320x320 with rain/glare/occlusion coverage.

    Deterministic cycle guarantees all three DR variants appear even for
    small n (sim-only, no Gazebo dependency).
    """
    rng = random.Random(seed)
    batch: list[dict] = []
    must = ["rain", "glare", "occlusion"]
    for i in range(n):
        if i < len(must) and n >= len(must):
            variant = must[i]
        else:
            variant = rng.choice(DR_VARIANTS)
        batch.append(
            {
                "id": i,
                "width": IMAGE_SIZE,
                "height": IMAGE_SIZE,
                "variant": variant,
                "source": SYNTH_NAME,
            }
        )
    return batch


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build dataset manifest + DR batch (sim-only)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "dataset_manifest.yaml"))
    ap.add_argument("--dry-run", action="store_true", help="validate without writing files")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    manifest = build_manifest_dict()
    batch = sample_batch(n=args.n, seed=args.seed)
    variants = sorted({b["variant"] for b in batch})
    print(
        f"sources={len(manifest['sources'])} total={sum(s['n'] for s in manifest['sources'])} "
        f"synthetic_ratio={manifest['synthetic_ratio']:.6f} batch={len(batch)} variants={variants}"
    )
    if args.dry_run:
        return 0
    generate_manifest(args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
