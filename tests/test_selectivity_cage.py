"""PR1 — Jaula virtual de selectividad SIMULATED (sim-only, 0 EUR).

Composicion-only sobre `replay_scenario` + `FusionPipeline.step_with_ai`.
No muta `tests/goldens/*.yaml` ni codigo B1-5 (solo lectura).

Cage: `bee_heavy` seed7 (100) + seed107 (100) -> N=200, bee_ratio 0.80,
5 frames/caso. Adverso: 4x12 rain/glare/occlusion/dusk.
"""
import numpy as np

from jetson.fusion.fusion_pipeline import FusionPipeline, fire_authorize
from sim.ros2.bag_replay import replay_scenario

K = [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]
R = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
T = [0.0, 0.0, 0.0]

# Truth-mapped Pv/Pb (determinista; mirrors test_fusion_ai).
# detector.predict stub capa Pv<=0.999 y es fg-dependiente: el harness
# usa este mapeo para una puerta estable y reproducible.
PV_PB = {"velutina": (0.997, 0.0004), "bee": (0.10, 0.40)}


def _pv_pb(truth: str) -> tuple:
    return PV_PB[truth]


def _velutina_mask():
    mask = np.zeros((320, 320), dtype=np.uint8)
    yy, xx = np.mgrid[0:320, 0:320]
    ellipse = ((xx - 160) / 60.0) ** 2 + ((yy - 160) / 25.0) ** 2 <= 1.0
    mask[ellipse] = 255
    return mask


def _bee_mask():
    # Foreground minimo (fg_ratio<0.005) -> feats bee-safe (R3/R4 fail).
    mask = np.zeros((320, 320), dtype=np.uint8)
    mask[10:20, 10:20] = 200
    return mask


def _run_case(case: dict, track_id: str) -> dict:
    """5x step_with_ai por caso; retorna {truth,promoted,Pv/Pb,vote,gates,shot}."""
    pipe = FusionPipeline(K=K, R=R, t=T, dt=0.1)
    pipe.initiate(case["z0"], case["z1"])
    pv, pb = _pv_pb(case["truth"])
    mask = _velutina_mask() if case["truth"] == "velutina" else _bee_mask()
    out = None
    for f in case["frames"]:
        lidar = None if f["lidar_xyz"] is None else np.asarray(f["lidar_xyz"])
        out = pipe.step_with_ai(
            lidar_xyz=lidar,
            v_radial=f["v_radial"],
            mask=mask,
            p_velutina=pv,
            p_bee=pb,
            track_id=track_id,
            longest_axis_mm=f["longest_axis_mm"],
            sideband_hz=f["sideband_hz"],
        )
    return {"truth": case["truth"], "promoted": bool(out["promoted"]),
            "Pv": pv, "Pb": pb, "vote": out.get("ai_vote"),
            "gates": dict(out["gates"]),
            "shot": fire_authorize(out, True, True)}


def _cage_dual_seed():
    return replay_scenario("bee_heavy", 7) + replay_scenario("bee_heavy", 107)


def test_cage_size_and_mix():
    cases = _cage_dual_seed()
    assert len(cases) >= 200, f"N={len(cases)} < 200"
    bee_ratio = sum(1 for c in cases if c["truth"] == "bee") / len(cases)
    assert bee_ratio >= 0.80, f"bee_ratio={bee_ratio:.3f} < 0.80"
    assert all(len(c["frames"]) == 5 for c in cases), "frames_per_case==5 requerido"


def test_cage_zero_bee_promotion():
    results = [_run_case(c, f"cage-{i}") for i, c in enumerate(_cage_dual_seed())]
    bees_promoted = sum(1 for r in results if r["truth"] == "bee" and r["promoted"])
    assert bees_promoted == 0, f"bees_promoted={bees_promoted} > 0"
    vel = [r for r in results if r["truth"] == "velutina"]
    assert len(vel) == 40, f"velutinas={len(vel)} != 40"
    assert all(r["vote"] is True for r in vel), "voto 2/3 debe promover velutina"


def test_cage_adverse_zero_bee_promotion():
    n = 0
    for variant in ("rain", "glare", "occlusion", "dusk"):
        cases = replay_scenario(variant, 11 if variant == "rain" else 7)
        assert len(cases) == 12, f"{variant}: {len(cases)} != 12"
        for i, c in enumerate(cases):
            r = _run_case(c, f"adv-{variant}-{i}")
            n += 1
            if r["truth"] == "bee":
                assert not r["promoted"], f"bee promovido en {variant} caso {i}"
    assert n == 48, f"adversos={n} != 48"


def test_cage_determinism_seed7_11():
    a = replay_scenario("bee_heavy", 7)
    b = replay_scenario("bee_heavy", 7)
    assert [c["truth"] for c in a] == [c["truth"] for c in b]
    assert a[0]["frames"][0]["lidar_xyz"] == b[0]["frames"][0]["lidar_xyz"]
    c = replay_scenario("rain", 11)
    d = replay_scenario("rain", 11)
    assert [f["lidar_xyz"] for f in c[0]["frames"]] == \
        [f["lidar_xyz"] for f in d[0]["frames"]]
    # Goldens solo lectura: base intacta, 100 casos / 80 bees.
    import pathlib
    golden = pathlib.Path("tests/goldens/bee_heavy.yaml").read_text()
    assert "bee_heavy" in golden and "total: 100" in golden
