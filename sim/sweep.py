"""Barrido de 1 parámetro (solo lectura, 0 euros, sin ventanas).

--eje distancia|lluvia|abejas|velocidad con 4-5 valores. Para cada valor
ejecuta la cadena real (FusionPipeline.step sobre casos de
replay_scenario) y muestra recall/FP en tabla ASCII.

No toca sim/ros2/bag_replay.py: compone desde fuera (genera los casos
con replay_scenario y los perturba/escala en este fichero).
"""

import argparse
import sys
from pathlib import Path

_RAIZ = Path(__file__).resolve().parents[1]
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))

import numpy as np

from jetson.ai import metrics as metrics_mod
from jetson.fusion.fusion_pipeline import FusionPipeline
from sim.ros2.bag_replay import replay_scenario

K = [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]
R = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
T = [0.0, 0.0, 0.0]

# Valores por defecto (4-5 por eje).
VALORES_DEFECTO = {
    "distancia": [1.0, 2.0, 3.0, 4.0, 5.0],       # metros (escala xyz)
    "lluvia": [0.0, 0.005, 0.010, 0.015, 0.025],  # sigma extra lidar (m)
    "abejas": [20, 40, 80, 120, 160],             # nº de abejas en la mezcla
    "velocidad": [1.0, 3.0, 5.0, 8.0, 11.0],      # m/s (escala v_radial)
}


def _corre_casos(casos):
    """Ejecuta la cadena y devuelve (recall, fp_abeja, precisión, n)."""
    yt, yp, sc = [], [], []
    for caso in casos:
        tubo = FusionPipeline(K=K, R=R, t=T, dt=0.1)
        tubo.initiate(caso["z0"], caso["z1"])
        fundido = None
        for marco in caso["frames"]:
            fundido = tubo.step(**marco)
        yt.append(1 if caso["truth"] == "velutina" else 0)
        yp.append(1 if bool(fundido["promoted"]) else 0)
        sc.append(float(fundido.get("conf", 0.0)))
    m = metrics_mod.compute(yt, yp, sc)
    return float(m["recall"]), float(m["bee_fp_rate"]), float(m["precision"])


def _escala_xyz(vec, factor):
    if vec is None:
        return None
    return (np.asarray(vec, dtype=float) * float(factor)).tolist()


def barrido(eje, valores=None, semilla=7):
    """Barrido externo: devuelve lista de dicts {valor, n, recall, fp, precision}."""
    eje = str(eje)
    if eje not in VALORES_DEFECTO:
        raise ValueError("eje debe ser uno de %s" % sorted(VALORES_DEFECTO))
    if valores is None:
        valores = list(VALORES_DEFECTO[eje])
    base = replay_scenario("bee_heavy", int(semilla))
    filas = []
    if eje == "abejas":
        # Mezcla: 40 velutinas fijas (2 jaulas) + N abejas (composición externa).
        pool = (replay_scenario("bee_heavy", int(semilla))
                + replay_scenario("bee_heavy", int(semilla) + 1000))
        velus = [c for c in pool if c["truth"] == "velutina"][:40]
        abejas = [c for c in pool if c["truth"] == "bee"]
        for v in valores:
            n_ab = max(0, int(v))
            mezcla = velus + abejas[:min(n_ab, len(abejas))]
            r, fp, p = _corre_casos(mezcla)
            filas.append({"valor": v, "n": len(mezcla),
                          "recall": r, "fp": fp, "precision": p})
        return filas
    for v in valores:
        casos = []
        rng = np.random.default_rng(int(semilla) + 999)
        for c in base:
            nc = {"truth": c["truth"], "variant": c["variant"],
                  "z0": list(c["z0"]), "z1": list(c["z1"]), "frames": []}
            if eje == "distancia":
                factor = float(v) / 3.0  # profundidad nominal 3 m
                nc["z0"] = _escala_xyz(c["z0"], factor)
                nc["z1"] = _escala_xyz(c["z1"], factor)
            if eje == "lluvia":
                nc["z0"] = list(c["z0"])
                nc["z1"] = list(c["z1"])
            for f in c["frames"]:
                nf = dict(f)
                nf["hsv"] = dict(f["hsv"])
                if eje == "distancia":
                    factor = float(v) / 3.0
                    nf["lidar_xyz"] = _escala_xyz(f["lidar_xyz"], factor)
                elif eje == "lluvia":
                    if f["lidar_xyz"] is None:
                        nf["lidar_xyz"] = None
                    else:
                        extra = rng.normal(0.0, float(v), 3) if float(v) > 0 else 0.0
                        nf["lidar_xyz"] = (
                            np.asarray(f["lidar_xyz"]) + extra).tolist()
                elif eje == "velocidad":
                    if f["v_radial"] is None:
                        nf["v_radial"] = None
                    else:
                        nf["v_radial"] = float(f["v_radial"]) * float(v) / 5.0
                nc["frames"].append(nf)
            casos.append(nc)
        r, fp, p = _corre_casos(casos)
        filas.append({"valor": v, "n": len(casos),
                      "recall": r, "fp": fp, "precision": p})
    return filas


def _imprime(eje, filas):
    print("BARRIDO eje=%s (perturbación externa, bag_replay.py intacto)" % eje)
    print("-" * 62)
    print("%-12s | %4s | %7s | %8s | %9s" % ("valor", "N", "recall", "FP abeja", "precisión"))
    print("-" * 62)
    for f in filas:
        print("%-12s | %4d | %7.4f | %8.5f | %9.4f" % (
            str(f["valor"]), f["n"], f["recall"], f["fp"], f["precision"]))
    print("-" * 62)
    print("Lectura: recall alto + FP abeja 0 es lo bueno (duda->abeja).")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Barrido de 1 parámetro")
    ap.add_argument("--eje", default="distancia",
                    choices=sorted(VALORES_DEFECTO),
                    help="parámetro a barrer")
    ap.add_argument("--valores", default=None,
                    help="lista separada por comas (si no, 4-5 por defecto)")
    ap.add_argument("--semilla", type=int, default=7)
    args = ap.parse_args(argv)
    if args.valores is None:
        valores = list(VALORES_DEFECTO[args.eje])
    else:
        valores = [s.strip() for s in str(args.valores).split(",") if s.strip()]
        # Acepta enteros o decimales según el eje.
        conv = int if args.eje == "abejas" else float
        valores = [conv(v) for v in valores]
    _imprime(args.eje, barrido(args.eje, valores, semilla=args.semilla))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
