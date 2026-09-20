"""RED test 2.4c — camera sim 120fps + projection (must fail before impl)."""
from sim.ros2.camera_sim import CameraSim

K = [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]


def test_projects_center_and_counts_frames():
    cam = CameraSim(K=K, fps=120)
    frame = cam.capture([0.0, 0.0, 2.0], target_class="velutina")
    assert abs(frame["u"] - 320.0) < 1e-9
    assert abs(frame["v"] - 240.0) < 1e-9
    for _ in range(119):
        cam.capture([0.0, 0.0, 2.0], target_class="velutina")
    assert cam.frames == 120
    assert cam.measured_fps(1.0) == 120  # 120 frames in 1 s sim


def test_velutina_vs_bee_hsv():
    cam = CameraSim(K=K, fps=120)
    vel = cam.capture([0.0, 0.0, 2.0], target_class="velutina")
    assert vel["thorax_v"] < 60
    assert 15 <= vel["band_h"] <= 45 and vel["band_s"] > 80
    bee = cam.capture([0.0, 0.0, 2.0], target_class="bee")
    assert bee["thorax_v"] >= 60  # amber-uniform → R3 fail downstream
