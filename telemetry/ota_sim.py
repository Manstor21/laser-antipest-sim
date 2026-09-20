"""OTA sim versionado metadata (SIMULATED, sim-only 0EUR).

Design #171: `telemetry/ota_sim.py` store dict {current, N-1, rejected};
`promote()` exige `export_onnx.check()` (si hay onnx_path) + goldens
rain/glare/occlusion con opset 12 e IoU>=0.80; fail->rollback N-1.
Goldens reutilizan `detector.predict+metrics.iou` (valores precomputados
en meta). 0EUR: sin binarios, sin GPU, sin subprocess, sin writes fuera
de memoria. Sim-time `t` explícito float.

Contratos:
  OtaSim(current="v1").promote(ver, meta, t) -> bool
  OtaSim.rollback(t) -> str (current tras rollback a N-1)
Meta: {"opset": 12, "goldens": {"rain","glare","occlusion"},
  optional "onnx_path": str}
"""

SIMULATED = True

OPSET_REQUIRED = 12
GOLDENS_REQUIRED = ("rain", "glare", "occlusion")
IOU_MIN = 0.80


class OtaSim:
    """Versionado metadata .onnx sim con gate check+goldens y N-1."""

    def __init__(self, current: str = "v1"):
        self.current = str(current)
        self.previous = str(current)  # N-1 (igual hasta primer promote ok)
        self.rejected: dict = {}
        self.log: list = []
        self._good = str(current)  # última versión estable (green gate)

    # -- gates ---------------------------------------------------------
    def _check_meta(self, meta: dict) -> tuple:
        if not isinstance(meta, dict):
            return (False, "meta must be dict")
        if int(meta.get("opset", -1)) != OPSET_REQUIRED:
            return (False,
                    f"opset must be {OPSET_REQUIRED}, "
                    f"got {meta.get('opset')}")
        goldens = meta.get("goldens", {})
        if not isinstance(goldens, dict):
            return (False, "goldens must be dict rain/glare/occlusion")
        for name in GOLDENS_REQUIRED:
            try:
                score = float(goldens[name])
            except (KeyError, TypeError, ValueError):
                return (False, f"golden {name} missing")
            if score < IOU_MIN:
                return (False,
                        f"golden {name} IoU {score:.3f} < {IOU_MIN:.2f} "
                        f"(occlusion)" if name == "occlusion"
                        else f"golden {name} IoU {score:.3f} < "
                        f"{IOU_MIN:.2f}")
        onnx_path = meta.get("onnx_path")
        if onnx_path is not None:
            try:
                from jetson.ai.export import export_onnx
                export_onnx.check(str(onnx_path))
            except Exception as e:
                return (False, f"onnx check failed: {e}")
        return (True, "ok")

    # -- ops -----------------------------------------------------------
    def promote(self, ver: str, meta: dict, t: float = 0.0) -> bool:
        """Promociona versión si gate green; si no, reject+log (bool)."""
        t = float(t)
        ver = str(ver)
        ok, reason = self._check_meta(meta)
        if not ok:
            self.rejected[ver] = {"reason": reason, "t": t}
            self.log.append({"ver": ver, "t": t, "result": "rejected",
                             "reason": reason})
            # Fail cerrado: current sigue en la última buena (N estable).
            self.current = self._good
            return False
        self.previous = self._good
        self.current = ver
        self._good = ver
        self.log.append({"ver": ver, "t": t, "result": "promoted",
                         "reason": "check+goldens green"})
        return True

    def rollback(self, t: float = 0.0) -> str:
        """Revierte a la última buena retenida (N-1 intacto)."""
        t = float(t)
        self.current = self._good
        self.log.append({"ver": self.current, "t": t, "result": "rollback",
                         "reason": f"rollback to N-1 {self.previous}"})
        return self.current
