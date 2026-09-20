"""PR2 RED — config copia validada (SIMULATED, sim-only 0EUR).

Spec: sdd/velutina-app-telemetry/spec #170 (app-config-sim).
Design: #171 — app/config_sim.py copia thresholds.yaml+fusion/config.yaml.
Tasks: #173 Phase 2 (2.3 RED, 2.4 config).

Sim-time t explícito N/A (config estática); SIMULATED=True.
Nunca escribe yaml real: checksums intactos.
"""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REAL_TH = ROOT / "jetson" / "ai" / "thresholds.yaml"
REAL_FUSION = ROOT / "jetson" / "fusion" / "config.yaml"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_config_simulated_flag():
    from app import config_sim

    assert config_sim.SIMULATED is True


def test_doubt_maps_to_bee_veto_stays_on():
    from app.config_sim import load_copy, validate

    cfg = load_copy()
    # Copia con observado bajo el mínimo real 0.995.
    verdict, errors = validate(cfg, p_velutina=0.60)
    assert verdict == "bee/doubt->bee"
    assert errors == []
    # Veto sigue on en la copia.
    assert cfg["thresholds"]["vote"]["bee_veto"] is True


def test_reject_p_min_1_2_and_veto_off():
    import copy

    from app.config_sim import load_copy, validate

    cfg = load_copy()
    bad = copy.deepcopy(cfg)
    bad["thresholds"]["p_velutina_min"] = 1.2
    verdict, errors = validate(bad)
    assert verdict == "reject"
    assert any("p_velutina_min" in e for e in errors)

    bad2 = copy.deepcopy(cfg)
    bad2["thresholds"]["vote"]["bee_veto"] = False
    verdict2, errors2 = validate(bad2)
    assert verdict2 == "reject"
    assert any("veto" in e.lower() for e in errors2)


def test_write_isolation_real_yaml_untouched(tmp_path):
    from app.config_sim import load_copy, save_copy, validate

    pre = (_sha(REAL_TH), _sha(REAL_FUSION))
    cfg = load_copy()
    verdict, _ = validate(cfg, p_velutina=0.60)
    assert verdict == "bee/doubt->bee"
    out = tmp_path / "thresholds_copy.yaml"
    save_copy(cfg, out)
    assert out.exists()
    # Real yamls intactos tras edits + save en copia.
    post = (_sha(REAL_TH), _sha(REAL_FUSION))
    assert post == pre
