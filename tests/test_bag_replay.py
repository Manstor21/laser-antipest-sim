"""RED test 4.2 — bag replay proof sintetica 60 s + sync<=1ms (must fail before impl)."""
import numpy as np

from sim.ros2.bag_replay import generate_sync_bag, replay_bag_to_pipeline
from sim.ros2.sync import compute_sync_rms_ms


def test_synthetic_60s_bag_sync_within_1ms():
    bag = generate_sync_bag(seconds=60, rate_hz=10, seed=7)
    assert bag["seconds"] == 60
    assert len(bag["offsets_ms"]) == 600  # 60 s @ 10 Hz
    rms = compute_sync_rms_ms(np.asarray(bag["offsets_ms"]))
    assert rms <= 1.0, f"RMS {rms:.3f} ms exceeds 1 ms budget"


def test_bag_replay_emits_fusion_tracks():
    bag = generate_sync_bag(seconds=2, rate_hz=10, seed=9)
    tracks = replay_bag_to_pipeline(bag, target_class="velutina")
    assert len(tracks) == len(bag["lidar_xyz"])
    last = tracks[-1]
    assert "x9" in last and "gates" in last and "roi" in last
    assert last["promoted"] is True
    bees = replay_bag_to_pipeline(bag, target_class="bee")
    assert bees[-1]["promoted"] is False  # doubt->bee holds on replay
