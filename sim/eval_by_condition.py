"""Evaluación por condición (solo lectura, 0 euros, sin ventanas).

Lee los escenarios dorados de sim/ros2/bag_replay.py (replay_scenario,
PERTURB) y los pasa por la cadena real FusionPipeline.step. Calcula con
jetson/ai/metrics.py (compute + iou): recall, FP abeja, precisión e IoU
de máscara. Imprime tabla por condición + matriz de confusión global.

Condiciones: bee_heavy + rain + glare + occlusion + dusk.
Todo determinista (semillas fijas). Solo usa stdlib + numpy (+ scipy si
está para nada crítico aquí).
"""

import sys
from pathlib import Path

# Raíz del proyecto para `python sim/eval_by_condition.py` directo.
_RAIZ = Path(__file__).resolve().parents[1]
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))

import numpy as np

from jetson.ai import detector as detector_mod
from jetson.ai import metrics as metrics_mod
from jetson.fusion.fusion_pipeline import FusionPipeline
from sim.ros2.bag_replay import replay_scenario

# Calibración de cámara igual que en los tests (solo lectura).
K = [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]
R = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
T = [0.0, 0.0, 0.0]

# Condiciones pedidas + semilla de cada golden (ver tests/goldens/*.yaml).
CONDICIONES = ["bee_heavy", "rain", "glare", "occlusion", "dusk"]
SEMILLA_BEE = 7
SEMILLA_ADV = 11


def _recorte_base():
    """Recorte brillante de referencia (igual que test_ai_goldens)."""
    base = np.zeros((320, 320, 3), dtype=np.uint8)
    base[140:180, 120:200, 0] = 200
    return base


def _recorte_condicion(nombre):
    """Recorte sintético por condición adversa (solo para IoU de máscara)."""
    base = _recorte_base()
    if nombre in ("bee_heavy", "nominal"):
        return base.copy()
    if nombre == "rain":
        rng = np.random.default_rng(11)
        ruido = (rng.random((320, 320, 3)) * 25).astype(np.uint8)
        return np.clip(base.astype(int) + ruido.astype(int), 0, 255).astype(np.uint8)
    if nombre == "glare":
        velo = np.zeros((320, 320, 3), dtype=np.uint8)
        velo[0:60, 0:320, :] = 40  # velo leve bajo el umbral del stub
        return np.clip(base.astype(int) + velo.astype(int), 0, 255).astype(np.uint8)
    if nombre == "occlusion":
        rec = base.copy()
        rec[150:180, 150:200, :] = 0  # esquina ocluida
        return rec
    if nombre == "dusk":
        # Atardecer: todo más oscuro pero el bicho sigue sobre el umbral.
        return (base.astype(float) * 0.6).astype(np.uint8)
    return base.copy()


def _iou_mascara(nombre):
    """IoU entre máscara de la condición y máscara base (detector real)."""
    base = _recorte_base()
    rec = _recorte_condicion(nombre)
    ref = detector_mod.predict(base)["mask"] > 0
    out = detector_mod.predict(rec)["mask"] > 0
    return float(metrics_mod.iou(out, ref))


def _evalua_casos(casos):
    """Pasa casos por FusionPipeline.step y devuelve y_true/y_pred/scores."""
    y_true, y_pred, scores = [], [], []
    tp = fn = fp = tn = 0
    for i, caso in enumerate(casos):
        tubo = FusionPipeline(K=K, R=R, t=T, dt=0.1)
        tubo.initiate(caso["z0"], caso["z1"])
        fundido = None
        for marco in caso["frames"]:
            fundido = tubo.step(**marco)
        verdad = 1 if caso["truth"] == "velutina" else 0
        pred = 1 if bool(fundido["promoted"]) else 0
        y_true.append(verdad)
        y_pred.append(pred)
        scores.append(float(fundido.get("conf", 0.0)))
        if verdad == 1 and pred == 1:
            tp += 1
        elif verdad == 1:
            fn += 1
        elif pred == 1:
            fp += 1
        else:
            tn += 1
    return y_true, y_pred, scores, {"TP": tp, "FN": fn, "FP": fp, "TN": tn}


def evalua(condiciones=None, semilla_bee=SEMILLA_BEE, semilla_adv=SEMILLA_ADV):
    """Evalúa cada condición y devuelve dict con métricas por condición.

    Devuelve:
      {"por_condicion": {nombre: {n, recall, fp_abeja, precision, iou,
                                  tp, fn, fp, tn}},
       "matriz": {"TP":..,"FN":..,"FP":..,"TN":..},
       "global": {recall, fp_abeja, precision}}
    """
    if condiciones is None:
        condiciones = list(CONDICIONES)
    por_cond = {}
    m_tp = m_fn = m_fp = m_tn = 0
    yt_all, yp_all, sc_all = [], [], []
    for nombre in condiciones:
        semilla = semilla_bee if nombre == "bee_heavy" else semilla_adv
        casos = replay_scenario(nombre, semilla)
        yt, yp, sc, cm = _evalua_casos(casos)
        m = metrics_mod.compute(yt, yp, sc)
        iou = _iou_mascara(nombre)
        por_cond[nombre] = {
            "n": len(casos),
            "recall": float(m["recall"]),
            "fp_abeja": float(m["bee_fp_rate"]),
            "precision": float(m["precision"]),
            "iou": float(iou),
            "tp": cm["TP"],
            "fn": cm["FN"],
            "fp": cm["FP"],
            "tn": cm["TN"],
        }
        m_tp += cm["TP"]
        m_fn += cm["FN"]
        m_fp += cm["FP"]
        m_tn += cm["TN"]
        yt_all += yt
        yp_all += yp
        sc_all += sc
    mg = metrics_mod.compute(yt_all, yp_all, sc_all)
    return {
        "por_condicion": por_cond,
        "matriz": {"TP": m_tp, "FN": m_fn, "FP": m_fp, "TN": m_tn},
        "global": {
            "recall": float(mg["recall"]),
            "fp_abeja": float(mg["bee_fp_rate"]),
            "precision": float(mg["precision"]),
        },
    }


def _imprime(res):
    print("EVALUACIÓN POR CONDICIÓN (cadena real + metrics.compute)")
    print("Semillas: bee_heavy=7, adversas=11. Métricas en tanto por uno.")
    print("-" * 78)
    print("%-10s | %4s | %7s | %8s | %9s | %6s" % (
        "condición", "N", "recall", "FP abeja", "precisión", "IoU"))
    print("-" * 78)
    for nombre in res["por_condicion"]:
        f = res["por_condicion"][nombre]
        print("%-10s | %4d | %7.4f | %8.5f | %9.4f | %6.3f" % (
            nombre, f["n"], f["recall"], f["fp_abeja"], f["precision"], f["iou"]))
    print("-" * 78)
    g = res["global"]
    print("GLOBAL: recall=%.4f  FP abeja=%.5f  precisión=%.4f" % (
        g["recall"], g["fp_abeja"], g["precision"]))
    cm = res["matriz"]
    print("")
    print("MATRIZ DE CONFUSIÓN GLOBAL (filas=verdad, columnas=predicho)")
    print("                pred velutina | pred abeja")
    print("  verdad velut. | %7d      | %7d" % (cm["TP"], cm["FN"]))
    print("  verdad abeja  | %7d      | %7d" % (cm["FP"], cm["TN"]))
    print("TP=%d FN=%d FP=%d TN=%d" % (cm["TP"], cm["FN"], cm["FP"], cm["TN"]))


if __name__ == "__main__":
    _imprime(evalua())
