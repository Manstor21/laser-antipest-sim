"""RED test 2.4a — Benewake-equiv LiDAR sim (must fail before impl)."""
import numpy as np


def test_hornet_at_3m_10hz_rmse():
    from sim.gazebo.benewake_equiv.plugin import BenewakeEquiv

    lidar = BenewakeEquiv(seed=3)
    ranges = []
    for _ in range(10):  # 1 s sim at 10 Hz
        r = lidar.scan([0.0, 0.0, 3.0])
        assert r is not None
        ranges.append(r)
    rmse = float(np.sqrt(np.mean((np.asarray(ranges) - 3.0) ** 2)))
    assert rmse <= 0.050, f"RMSE {rmse * 1000:.1f} mm > 50 mm"


def test_out_of_range_ignored():
    from sim.gazebo.benewake_equiv.plugin import BenewakeEquiv

    lidar = BenewakeEquiv(seed=3)
    assert lidar.scan([0.0, 0.0, 6.0]) is None  # beyond 5 m
    assert lidar.scan([0.0, 0.0, 0.3]) is None  # below 0.5 m
