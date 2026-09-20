"""RED 2.1 — PR2 OHEM 2-rondas ejecutable (debe fallar sin ohem.py)."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_mine_produces_hard_negatives_with_crabro_and_empty(tmp_path):
    from jetson.ai import ohem

    ckpt = tmp_path / "ckpt1.pt"
    ckpt.write_text("fake-ckpt-round1", encoding="utf-8")
    val = tmp_path / "val_bee_heavy.yaml"
    val.write_text(yaml.safe_dump({"val": "bee-heavy"}), encoding="utf-8")
    out = tmp_path / "hard_negatives.yaml"

    result = ohem.mine(str(ckpt), str(val), str(out))
    assert Path(result).exists(), "mine() must write hard_negatives.yaml"
    with open(result, encoding="utf-8") as f:
        hard = yaml.safe_load(f)
    items = hard.get("hard_negatives", hard if isinstance(hard, list) else [])
    assert isinstance(items, list) and len(items) >= 3, "need crabro+empty+close-bee"
    cats = {h.get("category") for h in items}
    assert "crabro" in cats, f"crabro FP must be mined: {cats}"
    assert "empty" in cats, f"empty station must be mined: {cats}"
    assert "close-bee" in cats, f"close-bee must be mined: {cats}"


def test_build_round2_reinjects_hard_negatives(tmp_path):
    from jetson.ai import ohem

    manifest = ROOT / "sim" / "dataset" / "dataset_manifest.yaml"
    assert manifest.exists(), f"missing PR1 manifest {manifest}"
    hard_p = tmp_path / "hard_negatives.yaml"
    with open(hard_p, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "hard_negatives": [
                    {"id": "crabro-001", "category": "crabro"},
                    {"id": "empty-001", "category": "empty"},
                    {"id": "closebee-001", "category": "close-bee"},
                ]
            },
            f,
        )
    out = tmp_path / "round2.yaml"
    result = ohem.build_round2(str(manifest), str(hard_p), str(out))
    assert Path(result).exists(), "build_round2() must write round2.yaml"
    with open(result, encoding="utf-8") as f:
        r2 = yaml.safe_load(f)
    assert "sources" in r2 or "datasets" in r2 or "train" in r2, "round2 must reference train data"
    blob = str(r2)
    assert "crabro" in blob and "empty" in blob, "round2 must re-inject mined FPs"
    assert r2.get("ohem_round") == 2 or "2" in blob, "round2 must be marked as OHEM round 2"
