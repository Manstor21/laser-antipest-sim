"""PR2 — Matriz de impacto en fauna SIMULATED (sim-only, 0 EUR).

bycatch==0 por construccion (bee_FP==0 del gate) sobre la jaula N=200
dual-seed 7+107. Exposicion acotada por veto R6+entorno+shutter via
`fire_authorize()` y caps duty<=0.1% / burst<=5+2 s. Composicion-only.
"""
import numpy as np

from jetson.fusion.fusion_pipeline import FusionPipeline, fire_authorize
from jetson.fusion.hard_rules import r6_interlock
from jetson.laser.fire_controller import BURST_MAX, DUTY_LIMIT, DutyMeter
from sim.ros2.bag_replay import replay_scenario

K = [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]
R = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
T = [0.0, 0.0, 0.0]
PV_PB = {"velutina": (0.997, 0.0004), "bee": (0.10, 0.40)}
SIMULATED = True
DISCLAIMER = "SIMULATED sin certificar (NO-CERT): bycatch==0 por construccion (FP=0)"


def _velutina_mask():
    mask = np.zeros((320, 320), dtype=np.uint8)
    yy, xx = np.mgrid[0:320, 0:320]
    mask[((xx - 160) / 60.0) ** 2 + ((yy - 160) / 25.0) ** 2 <= 1.0] = 255
    return mask


def _bee_mask():
    mask = np.zeros((320, 320), dtype=np.uint8)
    mask[10:20, 10:20] = 200
    return mask


def _run_case(case: dict, track_id: str) -> dict:
    pipe = FusionPipeline(K=K, R=R, t=T, dt=0.1)
    pipe.initiate(case["z0"], case["z1"])
    pv, pb = PV_PB[case["truth"]]
    mask = _velutina_mask() if case["truth"] == "velutina" else _bee_mask()
    out = None
    for f in case["frames"]:
        lidar = None if f["lidar_xyz"] is None else np.asarray(f["lidar_xyz"])
        out = pipe.step_with_ai(
            lidar_xyz=lidar, v_radial=f["v_radial"], mask=mask,
            p_velutina=pv, p_bee=pb, track_id=track_id,
            longest_axis_mm=f["longest_axis_mm"],
            sideband_hz=f["sideband_hz"])
    return {"truth": case["truth"], "promoted": bool(out["promoted"]),
            "shot": fire_authorize(out, True, True)}


def _cage_results():
    cases = replay_scenario("bee_heavy", 7) + replay_scenario("bee_heavy", 107)
    return [_run_case(c, f"fauna-{i}") for i, c in enumerate(cases)]


def _fauna_matrix(results: list) -> dict:
    vel = [r for r in results if r["truth"] == "velutina"]
    bee = [r for r in results if r["truth"] == "bee"]
    matrix = {
        "vespa_velutina": {"n": len(vel),
                           "promoted": sum(1 for r in vel if r["promoted"]),
                           "shot": sum(1 for r in vel if r["shot"]),
                           "effect": "target-suppression SIMULATED"},
        "apis_mellifera": {"n": len(bee),
                           "promoted": sum(1 for r in bee if r["promoted"]),
                           "shot": sum(1 for r in bee if r["shot"]),
                           "effect": "no-harm SIMULATED"},
    }
    matrix["bycatch"] = matrix["apis_mellifera"]["promoted"]
    return matrix


def _fauna_bycatch() -> int:
    return int(_fauna_matrix(_cage_results())["bycatch"])


def _veto_denies_shot() -> bool:
    out = {"promoted": True, "ai_vote": True,
           "gates": {"R1": True, "R2": True, "R3": True, "R4": True}}
    r6_blocked = not r6_interlock(human_min_m=1.0, track_mm=10.0)
    denied_r6 = fire_authorize(out, not r6_blocked, True) is False
    denied_env = fire_authorize(out, False, True) is False
    denied_shutter = fire_authorize(out, True, False) is False
    assert r6_blocked, "R6 debe inhibir con humano a 1 m"
    return bool(denied_r6 and denied_env and denied_shutter)


def test_fauna_matrix_zero_bee_bycatch():
    m = _fauna_matrix(_cage_results())
    assert m["apis_mellifera"]["promoted"] == 0, \
        f"bee promoted={m['apis_mellifera']['promoted']} > 0"
    assert m["apis_mellifera"]["shot"] == 0
    assert m["bycatch"] == 0
    assert _fauna_bycatch() == 0


def test_fauna_species_effect_matrix():
    m = _fauna_matrix(_cage_results())
    assert set(("vespa_velutina", "apis_mellifera")) <= set(m)
    assert m["vespa_velutina"]["n"] == 40 and m["apis_mellifera"]["n"] == 160
    assert m["vespa_velutina"]["promoted"] == 40, "velutina debe promoverse"
    assert "SIMULATED" in m["vespa_velutina"]["effect"]
    assert "SIMULATED" in m["apis_mellifera"]["effect"]


def test_fauna_veto_bounds_exposure():
    assert _veto_denies_shot() is True
    out = {"promoted": True, "ai_vote": True,
           "gates": {"R1": True, "R2": True, "R3": True, "R4": True}}
    log = []
    for human, track, env, shutter in ((1.0, 10.0, True, True),
                                       (5.0, 50.0, True, True),
                                       (5.0, 10.0, False, True)):
        r6_ok = r6_interlock(human, track)
        auth = fire_authorize(out, r6_ok and env, shutter)
        log.append({"human": human, "track": track, "r6_ok": r6_ok,
                    "auth": auth})
        assert auth is False, f"veto debe denegar: {log[-1]}"
    assert len(log) == 3 and all(not e["auth"] for e in log)


def test_fauna_duty_burst_caps_bound_exposure():
    dm = DutyMeter(limit=DUTY_LIMIT, window_s=1.0)
    t0 = 300.0
    for i in range(BURST_MAX):
        assert dm.request(t=t0 + i * 0.002, width_us=1.0) is True
    assert dm.duty(t0 + 0.010) <= DUTY_LIMIT
    m = _fauna_matrix(_cage_results())
    exposures = m["vespa_velutina"]["shot"] + m["apis_mellifera"]["shot"]
    assert exposures == m["vespa_velutina"]["promoted"] <= 40
    assert m["bycatch"] == 0


def test_fauna_simulated_disclaimer():
    assert SIMULATED is True
    assert "SIMULATED" in DISCLAIMER and "NO-CERT" in DISCLAIMER
    assert "bycatch==0" in DISCLAIMER
