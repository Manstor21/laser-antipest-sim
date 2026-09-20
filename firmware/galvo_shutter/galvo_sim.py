"""XY2-100 galvo 2nd-order sim (SIMULATED, 0EUR, no HW).

Per-axis discrete 2nd-order: acc = wn^2*(target-pos) - 2*zeta*wn*vel.
Defaults (wn=12000 rad/s, zeta=0.7, dt=20us) give ~476us 2% settle,
inside the 300-500us XY2-100 band. Abort if tracking error >10mm.
"""

import math

SIMULATED = True
SETTLE_BAND_MM = 0.5
ABORT_ERROR_MM = 10.0


class GalvoSim:
    """2-axis galvo simulator with settle/abort helpers."""

    def __init__(self, wn=12000.0, zeta=0.7, dt=20e-6):
        self.wn = float(wn)
        self.zeta = float(zeta)
        self.dt = float(dt)
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.tgt_x = 0.0
        self.tgt_y = 0.0
        self.t = 0.0
        self._t_set = 0.0
        self._settle_t = None
        self._settle_abs_t = None
        self._abort_abs_t = None

    def set_target(self, x_mm, y_mm):
        self.tgt_x = float(x_mm)
        self.tgt_y = float(y_mm)
        self._t_set = self.t
        self._settle_t = None
        self._settle_abs_t = None
        self._abort_abs_t = None
        return (self.tgt_x, self.tgt_y)

    def reset(self, x_mm=0.0, y_mm=0.0):
        self.pos_x = float(x_mm)
        self.pos_y = float(y_mm)
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.tgt_x = float(x_mm)
        self.tgt_y = float(y_mm)
        self._t_set = self.t
        self._settle_t = None
        self._settle_abs_t = None
        self._abort_abs_t = None
        return (self.pos_x, self.pos_y)

    def _axis_step(self, pos, vel, tgt):
        acc = self.wn**2 * (tgt - pos) - 2.0 * self.zeta * self.wn * vel
        vel += acc * self.dt
        pos += vel * self.dt
        return pos, vel

    def step(self):
        self.pos_x, self.vel_x = self._axis_step(self.pos_x, self.vel_x, self.tgt_x)
        self.pos_y, self.vel_y = self._axis_step(self.pos_y, self.vel_y, self.tgt_y)
        self.t += self.dt
        if self._settle_t is None and self.tracking_error_mm() <= SETTLE_BAND_MM:
            self._settle_t = self.t - self._t_set
            self._settle_abs_t = self.t
        if self.tracking_error_mm() > ABORT_ERROR_MM:
            self._abort_abs_t = self.t
        return (self.pos_x, self.pos_y)

    def tracking_error_mm(self):
        return math.hypot(self.tgt_x - self.pos_x, self.tgt_y - self.pos_y)

    def settled(self, band_mm=SETTLE_BAND_MM):
        return self.tracking_error_mm() <= float(band_mm)

    def settle_time_s(self):
        if self._settle_t is not None:
            return float(self._settle_t)
        return float(self.t - self._t_set)

    def abort_required(self, limit_mm=ABORT_ERROR_MM):
        return self.tracking_error_mm() > float(limit_mm)

    def settle_timestamp(self):
        """Absolute sim-time t when settle band first reached, or None."""
        if self._settle_abs_t is not None:
            return float(self._settle_abs_t)
        if self._settle_t is not None:
            return float(self._t_set + self._settle_t)
        if self.settled():
            return float(self.t)
        return None

    def abort_timestamp(self):
        """Absolute sim-time t of last tracking-error exceed, or None."""
        if self._abort_abs_t is not None:
            return float(self._abort_abs_t)
        if self.abort_required():
            return float(self.t)
        return None
