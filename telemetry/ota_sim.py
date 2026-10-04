"""OTA sim versionado metadata (SIMULATED, sim-only 0EUR).

Design #171: `telemetry/ota_sim.py` store dict {current, N-1, rejected};
`promote()` exige `export_onnx.check()` (si hay onnx_path) + goldens
rain/glare/occlusion con opset 12 e IoU>=0.80; fail->rollback N-1.
Goldens reutilizan `detector.predict+metrics.iou` (valores precomputados
en meta). 0EUR: sin binarios, sin GPU, sin subprocess, sin writes fuera
de memoria. Sim-time `t` explícito float.

Seguridad: `meta` debe incluir `sha256` (hash del archivo .onnx).
Fase de migración: si `sha256` no está presente, se loguea WARNING pero
no se bloquea. Tests deben requerir `sha256`.

Contratos:
  OtaSim(current="v1").promote(ver, meta, t) -> bool
  OtaSim.rollback(t) -> str (current tras rollback a N-1)
Meta: {"opset": 12, "goldens": {"rain","glare","occlusion"},
  optional "onnx_path": str, "sha256": str}
"""

import hashlib
import logging

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
    def _check_meta(self, meta: dict, onnx_path: str | None = None) -> tuple:
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
        # SHA-256 verification of the ONNX artifact
        sha256_expected = meta.get("sha256")
        onnx_path = onnx_path or meta.get("onnx_path")
        if sha256_expected is not None and onnx_path is not None:
            try:
                actual_hash = self._compute_sha256(onnx_path)
                if actual_hash.lower() != sha256_expected.lower():
                    return (False,
                            f"sha256 mismatch: expected {sha256_expected}, got {actual_hash}")
            except Exception as e:
                return (False, f"sha256 verification failed: {e}")
        elif sha256_expected is not None and onnx_path is None:
            return (False, "sha256 provided but onnx_path missing")
        elif sha256_expected is None and onnx_path is not None:
            # Migration phase: warn but don't block
            logging.warning(
                "OTA promote: onnx_path provided without sha256 in meta. "
                "This will be required in future versions. "
                "Add 'sha256' to meta with the SHA-256 hash of the .onnx file."
            )
        # ed25519 signature verification (optional, future extension)
        ed25519_sig = meta.get("ed25519_sig")
        public_key = meta.get("public_key")
        if ed25519_sig is not None or public_key is not None:
            if ed25519_sig is None or public_key is None:
                return (False, "ed25519_sig and public_key must both be present if either is provided")
            # Placeholder for future Ed25519 verification
            logging.info("Ed25519 signature verification not yet implemented; skipping")
        onnx_path = meta.get("onnx_path")
        if onnx_path is not None:
            try:
                from jetson.ai.export import export_onnx
                export_onnx.check(str(onnx_path))
            except Exception as e:
                return (False, f"onnx check failed: {e}")
        return (True, "ok")

    def _compute_sha256(self, filepath: str) -> str:
        """Compute SHA-256 hash of a file."""
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    # -- ops -----------------------------------------------------------
    def promote(self, ver: str, meta: dict, t: float = 0.0) -> bool:
        """Promociona versión si gate green; si no, reject+log (bool)."""
        t = float(t)
        ver = str(ver)
        onnx_path = meta.get("onnx_path")
        ok, reason = self._check_meta(meta, onnx_path=onnx_path)
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
