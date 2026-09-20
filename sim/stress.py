"""Pruebas de rotura (solo lectura, 0 euros, sin ventanas).

4 pruebas, cada una con criterio explícito PASS/FAIL:
 1. enjambre denso (100 insectos): sin excepciones + FP abeja == 0.
 2. LiDAR ausente (coast EKF): sin excepciones + el tubo devuelve x9.
 3. latencia alta (LinkSim baud bajo): sin excepciones + retardo lento
    mayor que el rápido + los paquetes llegan.
 4. shutter cerrado: sin excepciones + fire_authorize bloquea con
    shutter no listo + energía accesible < 1.8 uJ (OD>=3.5).
"""

import sys
import traceback
from pathlib import Path

_RAIZ = Path(__file__).resolve().parents[1]
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))

import numpy as np

from firmware.galvo_shutter.shutter_sim import ACCESSIBLE_LIMIT_J, ShutterSim
from jetson.fusion.fusion_pipeline import FusionPipeline, fire_authorize
from sim.link.link_sim import LinkSim
from sim.ros2.bag_replay import replay_scenario

K = [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]
R = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
T = [0.0, 0.0, 0.0]

try:
    from sim.demo_apiario import _mask_abeja as _demo_mask_abeja
    from sim.demo_apiario import _mask_velutina as _demo_mask_velutina
    from sim.demo_apiario import PV_PB as _DEMO_PV_PB

    def _mascara(tipo):
        # verdad bag_replay (bee/velutina) o demo (abeja).
        return _demo_mask_velutina() if tipo == "velutina" else _demo_mask_abeja()

    _PVPB = {"velutina": _DEMO_PV_PB["velutina"],
             "bee": _DEMO_PV_PB["abeja"],
             "abeja": _DEMO_PV_PB["abeja"]}
except Exception:
    _PVPB = {"velutina": (0.997, 0.0004), "bee": (0.10, 0.40), "abeja": (0.10, 0.40)}

    def _mascara(tipo):
        mask = np.zeros((320, 320), dtype=np.uint8)
        if tipo == "velutina":
            yy, xx = np.mgrid[0:320, 0:320]
            mask[((xx - 160) / 60.0) ** 2 + ((yy - 160) / 25.0) ** 2 <= 1.0] = 255
        else:
            mask[10:20, 10:20] = 200
        return mask


def _prueba_enjambre():
    """100 insectos (jaula bee_heavy seed 7): criterio FP==0 sin excepciones."""
    nombre = "enjambre denso (100 insectos)"
    criterio = "sin excepciones + FP abeja == 0"
    try:
        casos = replay_scenario("bee_heavy", 7)
        assert len(casos) == 100, "la jaula debe tener 100 casos"
        fp = 0
        for i, c in enumerate(casos):
            tubo = FusionPipeline(K=K, R=R, t=T, dt=0.1)
            tubo.initiate(c["z0"], c["z1"])
            pv, pb = _PVPB[c["truth"]]
            salida = None
            for f in c["frames"]:
                lidar = None if f["lidar_xyz"] is None else np.asarray(f["lidar_xyz"])
                salida = tubo.step_with_ai(
                    lidar_xyz=lidar, v_radial=f["v_radial"], mask=_mascara(c["truth"]),
                    p_velutina=float(pv), p_bee=float(pb),
                    track_id="stress-%d" % i,
                    longest_axis_mm=float(f["longest_axis_mm"]),
                    sideband_hz=float(f["sideband_hz"]))
            if c["truth"] == "bee" and bool(salida["promoted"]):
                fp += 1
        ok = (fp == 0)
        return {"nombre": nombre, "criterio": criterio, "pass": bool(ok),
                "detalle": "FP=%d (esperado 0), N=100" % fp}
    except Exception as exc:
        return {"nombre": nombre, "criterio": criterio, "pass": False,
                "detalle": "excepción: %s\n%s" % (exc, traceback.format_exc(limit=3))}


def _prueba_lidar_ausente():
    """Todo el LiDAR a None (coast EKF): criterio sin excepciones + hay x9."""
    nombre = "LiDAR ausente (coast EKF)"
    criterio = "sin excepciones + el tubo devuelve x9 (predicción)"
    try:
        casos = replay_scenario("occlusion", 11)[:4]
        n_ok = 0
        for c in casos:
            tubo = FusionPipeline(K=K, R=R, t=T, dt=0.1)
            tubo.initiate(c["z0"], c["z1"])
            salida = None
            for f in c["frames"]:
                # OJO: se fuerza dropout total desde fuera (no se toca bag_replay).
                salida = tubo.step(lidar_xyz=None, v_radial=f["v_radial"],
                                   hsv=dict(f["hsv"]),
                                   longest_axis_mm=float(f["longest_axis_mm"]),
                                   wingspan=float(f["wingspan"]),
                                   body=float(f["body"]),
                                   sideband_hz=float(f["sideband_hz"]))
            if isinstance(salida, dict) and len(salida.get("x9", [])) == 9:
                n_ok += 1
        ok = (n_ok == len(casos))
        return {"nombre": nombre, "criterio": criterio, "pass": bool(ok),
                "detalle": "coast OK en %d/%d tubos" % (n_ok, len(casos))}
    except Exception as exc:
        return {"nombre": nombre, "criterio": criterio, "pass": False,
                "detalle": "excepción: %s\n%s" % (exc, traceback.format_exc(limit=3))}


def _prueba_latencia():
    """LinkSim baud bajo: criterio retardo lento > rápido + entrega OK."""
    nombre = "latencia alta (LinkSim baud bajo)"
    criterio = "sin excepciones + retardo(9600) > retardo(115200) + recv entrega"
    try:
        rapido = LinkSim()  # UART115200 por defecto
        lento = LinkSim(baud=9600, prop_s=5e-6)
        d_rap = rapido.delay_for(32)
        d_len = lento.delay_for(32)
        env = lento.send(1.0, b"\x01" * 32)
        rec = lento.recv(float(env["t_recv"]))
        ok = (d_len > d_rap) and (len(rec) == 1)
        return {"nombre": nombre, "criterio": criterio, "pass": bool(ok),
                "detalle": "retardo rápido=%.6fs lento=%.6fs entregados=%d" % (
                    d_rap, d_len, len(rec))}
    except Exception as exc:
        return {"nombre": nombre, "criterio": criterio, "pass": False,
                "detalle": "excepción: %s\n%s" % (exc, traceback.format_exc(limit=3))}


def _prueba_shutter():
    """Shutter cerrado bloquea el disparo y atenúa OD>=3.5."""
    nombre = "shutter cerrado"
    criterio = "sin excepciones + authorize==False con shutter no listo + E_acc<1.8uJ"
    try:
        s = ShutterSim()
        s.command(close=True, t=0.0)  # cerrado (NC), sin alimentar watchdog
        listo = bool(s.evaluate(0.0))  # debe ser False (watchdog sin feed)
        acc = float(s.accessible_j(5e-3))  # pulso 5 mJ cerrado
        promovido = {"promoted": True, "ai_vote": True,
                     "gates": {"R1": True, "R2": True, "R3": True, "R4": True}}
        auth = bool(fire_authorize(promovido, True, False))  # shutter no listo
        ok = (not listo) and (not auth) and (acc < float(ACCESSIBLE_LIMIT_J))
        return {"nombre": nombre, "criterio": criterio, "pass": bool(ok),
                "detalle": "ready=%s authorize=%s E_acc=%.3eJ (límite %.1eJ)" % (
                    listo, auth, acc, float(ACCESSIBLE_LIMIT_J))}
    except Exception as exc:
        return {"nombre": nombre, "criterio": criterio, "pass": False,
                "detalle": "excepción: %s\n%s" % (exc, traceback.format_exc(limit=3))}


def pruebas_estres():
    """Ejecuta las 4 pruebas y devuelve dict {pruebas: [...], ok_total}."""
    pruebas = [_prueba_enjambre(), _prueba_lidar_ausente(),
               _prueba_latencia(), _prueba_shutter()]
    return {"pruebas": pruebas, "ok_total": all(p["pass"] for p in pruebas)}


def _imprime(res):
    print("PRUEBAS DE ROTURA (criterio explícito por prueba)")
    print("-" * 78)
    for p in res["pruebas"]:
        estado = "PASS" if p["pass"] else "FAIL"
        print("[%s] %s" % (estado, p["nombre"]))
        print("      criterio: %s" % p["criterio"])
        print("      detalle : %s" % p["detalle"])
    print("-" * 78)
    print("TOTAL: %s" % ("TODO PASS" if res["ok_total"] else "HAY FALLOS"))


if __name__ == "__main__":
    _imprime(pruebas_estres())
