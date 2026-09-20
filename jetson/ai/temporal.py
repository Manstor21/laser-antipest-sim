"""PR2 core: voto temporal 3 frames por track-id (2/3 + veto bee).

Promote iff >=min_agree frames con P(velutina)>=p_min en el mismo
track-id AND ningún frame supera p_bee_max (doubt->bee, no-fire).
"""
from __future__ import annotations


class TemporalVote:
    def __init__(
        self,
        p_min: float = 0.995,
        p_bee_max: float = 0.001,
        min_frames: int = 3,
        min_agree: int = 2,
        bee_veto: bool = True,
    ) -> None:
        self.p_min = p_min
        self.p_bee_max = p_bee_max
        self.min_frames = min_frames
        self.min_agree = min_agree
        self.bee_veto = bee_veto
        self._tracks: dict[str, list[tuple[float, float]]] = {}

    def update(self, track_id: str, p_velutina: float, p_bee: float) -> bool | None:
        buf = self._tracks.setdefault(str(track_id), [])
        buf.append((float(p_velutina), float(p_bee)))
        # Ventana deslizante de min_frames.
        if len(buf) > self.min_frames:
            buf.pop(0)
        if len(buf) < self.min_frames:
            return None
        if self.bee_veto and any(pb > self.p_bee_max for _, pb in buf):
            return False
        agree = sum(1 for pv, _ in buf if pv >= self.p_min)
        return True if agree >= self.min_agree else False

    def reset(self, track_id: str) -> None:
        self._tracks.pop(str(track_id), None)
