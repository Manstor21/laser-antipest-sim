"""9-DOF EKF tracker + 50 ms latency compensation (sim-only, zero hardware).

State: x = [x,y,z, vx,vy,vz, ax,ay,az]. Constant-acceleration model,
dt from sim timestamps. LiDAR observes position [x,y,z]; radar observes
radial velocity projected on the sensor line-of-sight (linearized EKF).
Pure numpy so pytest runs without a ROS2 installation.
"""

import numpy as np


class EKFTracker:
    """Extended Kalman filter for a single insect track."""

    def __init__(self, dt=0.1, q_acc=1e-4, r_pos=0.01, r_vel=0.1):
        self.dt = float(dt)
        self.x = np.zeros(9, dtype=float)
        self.P = np.eye(9, dtype=float)
        q = np.array([1e-8] * 3 + [1e-6] * 3 + [float(q_acc)] * 3)
        self.Q = np.diag(q)
        self.R_lidar = (float(r_pos) ** 2) * np.eye(3)
        self.R_radar = float(r_vel) ** 2

    def initiate(self, z0_xyz, z1_xyz):
        """Two-point track initiation: position + finite-difference velocity.

        Acc priors stay tight (cruise model); the loose 9-DOF covariance
        would otherwise amplify fix noise instead of smoothing it.
        """
        z0 = np.asarray(z0_xyz, dtype=float).reshape(3)
        z1 = np.asarray(z1_xyz, dtype=float).reshape(3)
        if z0.shape != (3,) or z1.shape != (3,):
            raise ValueError("init fixes must be xyz")
        self.x[:3] = z1
        self.x[3:6] = (z1 - z0) / self.dt
        self.x[6:] = 0.0
        self.P = np.diag([0.01 ** 2] * 3 + [0.2 ** 2] * 3 + [0.05 ** 2] * 3)
        return self.x.copy()

    @staticmethod
    def transition(dt):
        """Constant-acceleration transition matrix F(dt), 9x9."""
        F = np.eye(9, dtype=float)
        for i in range(3):
            F[i, 3 + i] = dt
            F[i, 6 + i] = 0.5 * dt * dt
            F[3 + i, 6 + i] = dt
        return F

    def predict(self, dt=None):
        """Advance state/covariance by dt (default sim step)."""
        F = self.transition(self.dt if dt is None else float(dt))
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        return self.x.copy()

    def predict_at(self, t_fire, t_meas):
        """Non-mutating predict to fire time from measurement timestamp.

        Uses dt=t_fire-t_meas so link jitter is compensated explicitly;
        never uses wall-clock now. Pure sim-time.
        """
        dt = float(t_fire) - float(t_meas)
        F = self.transition(dt)
        return (F @ self.x).copy()

    def predict_ms(self, ms):
        """Non-mutating +ms horizon prediction (latency compensation).

        Wrapper over predict_at(t_meas+dt, t_meas) for backward compat.
        """
        dt_s = float(ms) / 1000.0
        return self.predict_at(t_fire=dt_s, t_meas=0.0)

    def update_lidar(self, z_xyz):
        """Position fix from Benewake-equiv cluster centroid."""
        z = np.asarray(z_xyz, dtype=float).reshape(3)
        if z.shape != (3,):
            raise ValueError("lidar fix must be xyz")
        H = np.zeros((3, 9))
        H[:, :3] = np.eye(3)
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R_lidar
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(9) - K @ H) @ self.P
        return self.x.copy()

    def update_radar(self, v_radial, origin=(0.0, 0.0, 0.0)):
        """Radial-velocity fix; LOS linearized around current estimate."""
        o = np.asarray(origin, dtype=float).reshape(3)
        rel = self.x[:3] - o
        r = float(np.linalg.norm(rel))
        if r < 1e-6:
            raise ValueError("track coincides with radar origin")
        los = rel / r
        vel = self.x[3:6]
        vr_pred = float(los @ vel)
        H = np.zeros((1, 9))
        H[0, :3] = (vel - vr_pred * los) / r
        H[0, 3:6] = los
        y = np.array([float(v_radial) - vr_pred])
        S = H @ self.P @ H.T + self.R_radar
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y).reshape(9)
        self.P = (np.eye(9) - K @ H) @ self.P
        return self.x.copy()
