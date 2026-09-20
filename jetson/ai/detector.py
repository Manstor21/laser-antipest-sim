"""PR2 core: inferencia ONNX CPU sobre ROI 320 (sim-only, 0EUR).

`predict(crop)` solo acepta crop 320x320 (roi_manager.get_roi());
devuelve {mask (320,320), p_velutina, p_bee}. Stub numpy determinista —
el cableado ONNX real vive en PR3 export (opset12).
"""
from __future__ import annotations

from typing import Dict

import numpy as np

ROI = 320


def _check_roi(crop: np.ndarray) -> None:
    if not isinstance(crop, np.ndarray):
        raise ValueError("crop must be numpy array 320x320x3")
    if crop.shape != (ROI, ROI, 3) and crop.shape != (ROI, ROI):
        raise ValueError(f"ROI-only 320x320 required, got {crop.shape}")


def predict(crop: np.ndarray) -> Dict[str, object]:
    """Predice mask + P(velutina)/P(bee) desde crop ROI 320."""
    _check_roi(crop)
    if crop.ndim == 3:
        gray = crop.mean(axis=2).astype(np.float64)
    else:
        gray = crop.astype(np.float64)
    # Mask sim: píxeles brillantes (thorax/bandas) -> foreground.
    mask = (gray > 100).astype(np.uint8) * 255
    fg_ratio = float((mask > 0).mean())
    # Scores deterministas desde energía del crop (stub calibrado en PR2/3).
    p_velutina = float(min(0.999, max(0.001, 0.10 + 0.85 * fg_ratio)))
    p_bee = float(min(0.90, max(0.0001, 0.30 * (1.0 - fg_ratio) + 0.02 * fg_ratio)))
    # Normalización suave: ambas son salidas independientes del head seg/cls.
    return {"mask": mask, "p_velutina": p_velutina, "p_bee": p_bee}


def main(argv=None) -> int:
    """Harness runtime sim: --roi-golden corre predict sobre ROI 320 sintético."""
    import argparse

    ap = argparse.ArgumentParser(description="Detector ROI 320 sim-only")
    ap.add_argument("--roi-golden", action="store_true")
    args = ap.parse_args(argv)
    crop = np.zeros((ROI, ROI, 3), dtype=np.uint8)
    crop[140:180, 120:200, 0] = 200
    out = predict(crop)
    print(f"roi-golden mask={out['mask'].shape} "
          f"Pv={out['p_velutina']:.4f} Pb={out['p_bee']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
