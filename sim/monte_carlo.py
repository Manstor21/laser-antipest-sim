"""Monte Carlo de la jaula virtual (solo lectura, 0 euros, sin ventanas).

Repite la jaula bee_heavy con N semillas y resume recall, FP abeja y
precisión como media ± desviación + IC95%.

Composición sobre sim/demo_apiario.py: importa PV_PB y máscaras de allí
(sin copiar código). Si el import fallase, usa un ayudante local mínimo
equivalente (definido abajo, sin tocar el original).
"""

import argparse
import math
import sys
from pathlib import Path

_RAIZ = Path(__file__).resolve().parents[1]
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))

import numpy as np

from jetson.ai import metrics as metrics_mod
from jetson.fusion.fusion_pipeline import FusionPipeline, fire_authorize
from sim.ros2.bag_replay import replay_scenario

K = [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]
R = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
T = [0.0, 0.0, 0.0]

# --- Composición: reutiliza lo de demo_apiario sin copiar ni modificar. ---
try:
    from sim.demo_apiario import PV_PB as _DEMO_PV_PB  # noqa: F401
    from sim.demo_apiario import _mask_abeja as _demo_mask_abeja
    from sim.demo_apiario import _mask_velutina as _demo_mask_velutina

    PV_PB = {"velutina": _DEMO_PV_PB["velutina"],
             "bee": _DEMO_PV_PB["abeja"],
             "abeja": _DEMO_PV_PB["abeja"]}

    def _mascara(tipo):
        # tipo puede ser verdad bag_replay (bee/velutina) o demo (abeja).
        if tipo == "velutina":
            return _demo_mask_velutina()
        return _demo_mask_abeja()

    _ORIGEN_MASCARAS = "sim/demo_apiario.py (importado, sin copiar)"
except Exception:
    # Ayudante local mínimo (solo si el import no es posible sin efectos).
    PV_PB = {"velutina": (0.997, 0.0004), "abeja": (0.10, 0.40)}

    def _mascara(tipo):
        mask = np.zeros((320, 320), dtype=np.uint8)
        if tipo == "velutina":
            yy, xx = np.mgrid[0:320, 0:320]
            elipse = ((xx - 160) / 60.0) ** 2 + ((yy - 160) / 25.0) ** 2 <= 1.0
            mask[elipse] = 255
        else:
            mask[10:20, 10:20] = 200
        return mask

    _ORIGEN_MASCARAS = "ayudante local (demo_apiario no importable)"


def _jaula_una_semilla(semilla):
    """Una jaula bee_heavy completa con voto IA temporal (como la demo)."""
    casos = replay_scenario("bee_heavy", int(semilla))
    yt, yp, sc = [], [], []
    for i, caso in enumerate(casos):
        tubo = FusionPipeline(K=K, R=R, t=T, dt=0.1)
        tubo.initiate(caso["z0"], caso["z1"])
        pv, pb = PV_PB[caso["truth"]]
        mask = _mascara(caso["truth"])
        salida = None
        for f in caso["frames"]:
            lidar = None if f["lidar_xyz"] is None else np.asarray(f["lidar_xyz"])
            salida = tubo.step_with_ai(
                lidar_xyz=lidar,
                v_radial=f["v_radial"],
                mask=mask,
                p_velutina=float(pv),
                p_bee=float(pb),
                track_id="mc-%d-%d" % (semilla, i),
                longest_axis_mm=float(f["longest_axis_mm"]),
                sideband_hz=float(f["sideband_hz"]),
            )
        yt.append(1 if caso["truth"] == "velutina" else 0)
        yp.append(1 if bool(salida["promoted"]) else 0)
        sc.append(float(salida.get("conf", 0.0)))
    m = metrics_mod.compute(yt, yp, sc)
    return {"recall": float(m["recall"]), "fp": float(m["bee_fp_rate"]),
            "precision": float(m["precision"])}


def _z95():
    """Cuantil 97.5% normal: scipy si está, si no 1.96 (fórmula normal)."""
    try:
        from scipy import stats
        return float(stats.norm.ppf(0.975))
    except Exception:
        return 1.96


def _resume(vals):
    """Media ± desviación (ddof=1) + IC95% normal."""
    n = len(vals)
    media = float(np.mean(vals)) if n else 0.0
    desv = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    z = _z95()
    semi = z * desv / math.sqrt(n) if n > 0 else 0.0
    return {"media": media, "desv": desv,
            "ic_inf": media - semi, "ic_sup": media + semi}


def monte_carlo(n_semillas=30, semilla_base=7):
    """Repite la jaula N veces y devuelve resumen estadístico."""
    recalls, fps, precs = [], [], []
    for k in range(int(n_semillas)):
        r = _jaula_una_semilla(int(semilla_base) + k)
        recalls.append(r["recall"])
        fps.append(r["fp"])
        precs.append(r["precision"])
    return {
        "n": int(n_semillas),
        "semilla_base": int(semilla_base),
        "origen_mascaras": _ORIGEN_MASCARAS,
        "recall": _resume(recalls),
        "fp": _resume(fps),
        "precision": _resume(precs),
    }


def _imprime(res):
    print("MONTE CARLO jaula bee_heavy: %d semillas desde %d" % (
        res["n"], res["semilla_base"]))
    print("Máscaras: %s" % res["origen_mascaras"])
    print("-" * 74)
    print("%-10s | %8s | %8s | %21s" % ("métrica", "media", "±desv", "IC95%"))
    print("-" * 74)
    for nombre in ("recall", "fp", "precision"):
        f = res[nombre]
        print("%-10s | %8.5f | %8.5f | [%.5f, %.5f]" % (
            nombre, f["media"], f["desv"], f["ic_inf"], f["ic_sup"]))
    print("-" * 74)
    print("IC95%% normal: media +- %.3f*desv/sqrt(N)" % (_z95(),))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Monte Carlo jaula virtual")
    ap.add_argument("--semillas", type=int, default=30,
                    help="número de semillas (defecto 30)")
    ap.add_argument("--base", type=int, default=7,
                    help="semilla base (defecto 7)")
    args = ap.parse_args(argv)
    _imprime(monte_carlo(n_semillas=args.semillas, semilla_base=args.base))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
