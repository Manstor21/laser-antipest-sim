"""MEMS mic stub — reserved topic only, zero DSP (R6 reserved).

Constraint: sim-only slice exposes `mems_mic/audio_raw` so future phases can
bind without code churn, but MUST NOT implement any audio processing.
"""

RESERVED_TOPIC = "mems_mic/audio_raw"
DSP_ENABLED = False


class MicStub:
    """Reserved microphone endpoint. Connects, returns zero audio."""

    def __init__(self) -> None:
        self.is_connected = False
        self.processed_samples = 0

    def connect(self) -> bool:
        self.is_connected = True
        return True

    def read(self, num_frames: int = 0):
        """Return zero audio frames; never touches DSP."""
        if not self.is_connected:
            raise RuntimeError("stub not connected")
        return []

    def process(self, samples):
        """Always refused — DSP is out of scope for the sim-only slice."""
        raise NotImplementedError("mems_mic DSP is reserved (R6 stub only)")
