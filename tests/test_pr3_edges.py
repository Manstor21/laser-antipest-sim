"""PR3 edge guards — pipeline/bag_replay robustness (REFACTOR, sim-only)."""
import numpy as np
import pytest

from jetson.fusion.fusion_pipeline import FusionPipeline
from sim.ros2.bag_replay import replay_scenario

K = [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]
R = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
T = [0.0, 0.0, 0.0]
HSV = {"thorax_v": 40.0, "band_h": 30.0, "band_s": 120.0}


def test_step_before_initiate_raises():
    pipe = FusionPipeline(K=K, R=R, t=T)
    with pytest.raises(RuntimeError):
        pipe.step(lidar_xyz=[1.0, 0.0, 3.0], v_radial=0.0, hsv=HSV)


def test_degenerate_radar_geometry_coasts():
    pipe = FusionPipeline(K=K, R=R, t=T)
    pipe.initiate([0.0, 0.0, 0.1], [0.0, 0.0, 0.1])
    # origin == track position -> update_radar raises ValueError internally,
    # pipeline must coast instead of crashing.
    fused = pipe.step(lidar_xyz=[0.0, 0.0, 0.1], v_radial=5.0, hsv=HSV,
                      longest_axis_mm=30.0, wingspan=40.0, body=20.0,
                      sideband_hz=130.0, origin=(0.0, 0.0, 0.1))
    assert "x9" in fused


def test_hist_truncation_at_max():
    pipe = FusionPipeline(K=K, R=R, t=T, hist_max=3)
    pipe.initiate([0.0, 0.0, 3.0], [0.5, 0.0, 3.0])
    for i in range(6):
        pipe.step(lidar_xyz=[0.5 + 0.1 * i, 0.0, 3.0], v_radial=1.0, hsv=HSV)
    assert len(pipe.vel_hist) == 3


def test_unknown_scenario_raises():
    with pytest.raises(ValueError):
        replay_scenario("nocturna", seed=0)
