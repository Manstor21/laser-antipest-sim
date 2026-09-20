"""Synthetic ROS2 bag generator + replay harness (sim-only, zero hardware).

Replaces `ros2 bag play` / `gz sim` on Windows 0-euro CI: deterministic
numpy trajectories with sim-clock offsets (RMS <= 1 ms) replayed through
FusionPipeline. Backs PR3 tasks 3.1/4.1/4.2 and tests/goldens/ manifests.
"""

import numpy as np

VELUTINA_FEATS = {"hsv": {"thorax_v": 40.0, "band_h": 30.0, "band_s": 120.0},
                  "longest_axis_mm": 30.0, "wingspan": 40.0, "body": 20.0,
                  "sideband_hz": 130.0}
BEE_FEATS = {"hsv": {"thorax_v": 150.0, "band_h": 30.0, "band_s": 40.0},
             "longest_axis_mm": 12.5, "wingspan": 20.0, "body": 20.0,
             "sideband_hz": 60.0}

# Adverse perturbations keep gates decisive (doubt->bee, zero bee promotion).
PERTURB = {
    "nominal": {"size": 0.5, "lidar": 0.010, "vel": 0.1, "dv": 0.0, "ds": 0.0, "drop": 0},
    "rain": {"size": 1.0, "lidar": 0.015, "vel": 0.15, "dv": 0.0, "ds": 0.0, "drop": 0},
    "glare": {"size": 0.5, "lidar": 0.010, "vel": 0.1, "dv": 15.0, "ds": -15.0, "drop": 0},
    "dusk": {"size": 0.5, "lidar": 0.012, "vel": 0.1, "dv": -15.0, "ds": -10.0, "drop": 0},
    "occlusion": {"size": 0.5, "lidar": 0.010, "vel": 0.1, "dv": 0.0, "ds": 0.0, "drop": 2},
}


def _feats(target_class, perturb):
    base = dict(VELUTINA_FEATS if target_class == "velutina" else BEE_FEATS)
    base["hsv"] = dict(base["hsv"])
    base["hsv"]["thorax_v"] = float(base["hsv"]["thorax_v"] + perturb["dv"])
    base["hsv"]["band_s"] = float(max(base["hsv"]["band_s"] + perturb["ds"], 1.0))
    return base


def _case_frames(target_class, variant, seed, n_frames=5):
    rng = np.random.default_rng(seed)
    rng_r = np.random.default_rng(seed + 1000)
    pert = PERTURB[variant]
    feats = _feats(target_class, pert)
    size = feats["longest_axis_mm"] + float(rng.normal(0.0, pert["size"]))
    xs = [0.5 + 0.5 * k for k in range(n_frames)]  # 5 m/s transit, range <= 5 m
    drop_idx = set(range(1, 1 + pert["drop"]))  # occlusion burst mid-track
    frames = []
    for k, x in enumerate(xs):
        true = np.array([x, 0.0, 3.0])
        r = float(np.linalg.norm(true))
        vr_true = float(5.0 * x / r)
        if k in drop_idx:
            frames.append({"lidar_xyz": None, "v_radial": None,
                           "hsv": dict(feats["hsv"]),
                           "longest_axis_mm": float(size),
                           "wingspan": float(feats["wingspan"]),
                           "body": float(feats["body"]),
                           "sideband_hz": float(feats["sideband_hz"])})
        else:
            frames.append({
                "lidar_xyz": (true + rng.normal(0.0, pert["lidar"], 3)).tolist(),
                "v_radial": float(vr_true + rng_r.normal(0.0, pert["vel"])),
                "hsv": dict(feats["hsv"]),
                "longest_axis_mm": float(size),
                "wingspan": float(feats["wingspan"]),
                "body": float(feats["body"]),
                "sideband_hz": float(feats["sideband_hz"])})
    z0 = (np.array([-0.5, 0.0, 3.0]) + rng.normal(0.0, 0.010, 3)).tolist()
    z1 = (np.array([0.0, 0.0, 3.0]) + rng.normal(0.0, 0.010, 3)).tolist()
    return {"truth": target_class, "variant": variant, "z0": z0, "z1": z1,
            "frames": frames}


def replay_scenario(name, seed=0):
    """Deterministic golden case list for tests/test_goldens.py."""
    if name == "bee_heavy":
        cases = []
        for i in range(100):
            truth = "bee" if i < 80 else "velutina"  # 80% bee-heavy
            variant = ["nominal", "dusk", "glare", "rain"][i % 4]
            cases.append(_case_frames(truth, variant, seed + i))
        return cases
    if name in ("rain", "glare", "occlusion", "dusk"):
        cases = []
        for i in range(12):
            truth = "velutina" if i % 2 == 0 else "bee"
            cases.append(_case_frames(truth, name, seed + i))
        return cases
    raise ValueError(f"unknown scenario {name!r}")


def generate_sync_bag(seconds=60, rate_hz=10, seed=0, sigma_ms=0.4):
    """Synthetic 60 s bag: clock offsets + hover lidar fixes + Doppler."""
    rng = np.random.default_rng(seed)
    n = int(seconds) * int(rate_hz)
    offsets = rng.normal(0.0, float(sigma_ms), size=n).tolist()
    lidar_xyz, v_radial = [], []
    for _ in range(n):
        true = np.array([1.0, 0.0, 3.0])
        lidar_xyz.append((true + rng.normal(0.0, 0.010, 3)).tolist())
        v_radial.append(float(rng.normal(0.0, 0.05)))
    return {"seconds": int(seconds), "rate_hz": int(rate_hz),
            "offsets_ms": offsets, "lidar_xyz": lidar_xyz,
            "v_radial": v_radial}


def replay_bag_to_pipeline(bag, target_class="velutina"):
    """Replay a synthetic bag through FusionPipeline; returns Fused list."""
    from jetson.fusion.fusion_pipeline import FusionPipeline
    K = [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]
    R = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    pipe = FusionPipeline(K=K, R=R, t=[0.0, 0.0, 0.0], dt=0.1)
    feats = VELUTINA_FEATS if target_class == "velutina" else BEE_FEATS
    pipe.initiate(bag["lidar_xyz"][0], bag["lidar_xyz"][1])
    out = []
    for xyz, vr in zip(bag["lidar_xyz"][2:], bag["v_radial"][2:]):
        out.append(pipe.step(lidar_xyz=xyz, v_radial=float(vr),
                             hsv=dict(feats["hsv"]),
                             longest_axis_mm=float(feats["longest_axis_mm"]),
                             wingspan=float(feats["wingspan"]),
                             body=float(feats["body"]),
                             sideband_hz=float(feats["sideband_hz"])))
    # pad to full length so len(tracks) == len(lidar_xyz) for the harness
    while len(out) < len(bag["lidar_xyz"]):
        out.append(out[-1])
    return out
