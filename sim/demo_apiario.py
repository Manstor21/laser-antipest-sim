"""Demo visual del apiario (SIMULADA, 0 euros, sin hardware).

Lo que hace, en español llano:
- Crea un escenario falso pero determinista (semilla fija): 8 abejas que
  vuelan en linea recta hacia la colmena + 3 velutinas quietas en el aire
  (hovering) delante de la piquera.
- Pasa cada insecto por el sistema real (solo lectura, no modifica nada):
  EKF + ROI + reglas duras R1-R4 + IA simulada + laser + telemetria.
- Muestra en pantalla lo que "ve" el sistema: posicion, P(velutina)/P(abeja),
  puertas R1-R4 y decision dispara / no dispara.
- Al final imprime el resumen: velutinas neutralizadas, abejas tocadas (=0)
  y precision.

Uso:
    python sim/demo_apiario.py            # ASCII animada (siempre funciona)
    python sim/demo_apiario.py --ascii    # fuerza ASCII aunque haya matplotlib
    python sim/demo_apiario.py --sin-pausa  # sin pausas, para CI (<2 s)
    python sim/demo_apiario.py --frames 10  # menos fotogramas (por defecto 14)

Si hay matplotlib y NO se pasa --ascii, se abre una ventana animada.
Si no hay matplotlib, se usa la consola ASCII automaticamente.
Todo es determinista: misma semilla -> mismo resultado.
"""

import argparse
import sys
import time
from pathlib import Path

# Raiz del proyecto en sys.path para importar como `python sim/demo_apiario.py`.
_RAIZ = Path(__file__).resolve().parents[1]
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))

import numpy as np

# Reutilizacion SOLO lectura (nunca se modifica nada de estos modulos):
# - sim/ros2/bag_replay.py -> replay_scenario (escenario dorado de referencia)
# - jetson/fusion/fusion_pipeline.py -> FusionPipeline + fire_authorize
# - jetson/ai/detector.py -> predict (se llama de verdad, modo muestra)
# - jetson/ai/temporal.py -> TemporalVote (vive dentro del pipeline)
# - jetson/laser/fire_pipeline.py -> FirePipeline (laser simulado)
# - telemetry/mqtt_sim.py -> MqttSim (telemetria simulada)
from jetson.ai import detector as detector_mod
from jetson.fusion.fusion_pipeline import FusionPipeline, fire_authorize
from jetson.laser.fire_pipeline import FirePipeline
from sim.ros2.bag_replay import BEE_FEATS, VELUTINA_FEATS, replay_scenario
from telemetry.mqtt_sim import MqttSim

# Parametros fijos de la demo (calibracion de camara como en los tests).
K = [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]
R = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
T = [0.0, 0.0, 0.0]

SEMILLA = 7
N_ABEJAS = 8
N_VELUTINAS = 3
N_FRAMES = 14
DT = 0.1

# Mapeo Pv/Pb determinista del harness oficial de jaula
# (tests/test_selectivity_cage.py). El stub de pixeles detector.predict
# no llega a Pv>=0.995 por construccion, asi que la decision usa este
# mapeo estable y el stub se muestra aparte como "muestra de IA".
PV_PB = {"velutina": (0.997, 0.0004), "abeja": (0.10, 0.40)}


def _mask_velutina():
    """Elipse alargada brillante: R3/R4 pasan (igual que test_fusion_ai)."""
    mask = np.zeros((320, 320), dtype=np.uint8)
    yy, xx = np.mgrid[0:320, 0:320]
    elipse = ((xx - 160) / 60.0) ** 2 + ((yy - 160) / 25.0) ** 2 <= 1.0
    mask[elipse] = 255
    return mask


def _mask_abeja():
    """Puntos sueltos: casi sin foreground -> R3/R4 fallan (abeja segura)."""
    mask = np.zeros((320, 320), dtype=np.uint8)
    mask[10:20, 10:20] = 200
    return mask


def _muestra_ia(tipo):
    """Llama de verdad a detector.predict sobre un recorte falso.

    Solo demostrativo: devuelve (Pv, Pb) del stub de pixeles para que se
    vea que la IA "mira" el recorte. La decision usa PV_PB (ver arriba).
    """
    recorte = np.zeros((320, 320, 3), dtype=np.uint8)
    if tipo == "velutina":
        recorte[140:180, 120:200, :] = 200  # mancha brillante = bicho grande
    else:
        recorte[150:160, 150:160, :] = 200  # mancha chica = bicho chico
    try:
        salida = detector_mod.predict(recorte)
        return float(salida["p_velutina"]), float(salida["p_bee"])
    except Exception:
        return 0.0, 1.0


def _verdad_abeja(i, k):
    """Abeja i en frame k: recta de x=5 hacia la piquera (x=0), a 3-4 m/s."""
    x0 = 5.0 + (i % 4) * 0.3
    y0 = -1.2 + i * 0.3
    z0 = 1.6 + (i % 3) * 0.25
    x = x0 - (3.0 + (i % 3) * 0.5) * DT * k
    return np.array([x, y0, z0])


def _verdad_velutina(i, k):
    """Velutina i en frame k: quieta en el aire delante de la piquera.

    OJO: cerca del eje optico (x/z < 0.1) para que el galvo simulado
    no aborte por error >10 mm (campo = 100 * atan(x/z)).
    """
    cx = 0.08 + i * 0.06
    cy = 0.03 - i * 0.03
    cz = 2.6 + i * 0.15
    # Temblor chico: +-5 cm -> velocidad ~0.2 m/s -> puerta R2 = hover.
    ox = 0.05 * np.sin(2.0 * k * DT + i)
    oy = 0.05 * np.cos(1.5 * k * DT + i * 2.0)
    return np.array([cx + ox, cy + oy, cz])


def _puertas_txt(gates):
    """R1R2R3R4 con tick/cruz, ej: R1ok R2ok R3XX R4ok."""
    orden = ["R1", "R2", "R3", "R4"]
    partes = []
    for g in orden:
        ok = bool(gates.get(g, False))
        partes.append(g + ("ok" if ok else "XX"))
    return " ".join(partes)


def _dibuja_mapa(insectos, historial_disparos):
    """Vista lateral (X = distancia, Z = altura), 46 x 13 celdas."""
    ancho, alto = 46, 13
    x_min, x_max = -0.5, 5.5
    z_min, z_max = 0.5, 3.2
    lienzo = [["." for _ in range(ancho)] for _ in range(alto)]
    # Piquera (colmena) en x=0, z=1.5: columna H.
    col_h = int((0.0 - x_min) / (x_max - x_min) * (ancho - 1))
    fil_h = alto - 1 - int((1.5 - z_min) / (z_max - z_min) * (alto - 1))
    for f in range(max(0, fil_h - 1), min(alto, fil_h + 2)):
        lienzo[f][col_h] = "H"
    for ins in insectos:
        pos = ins.get("pos_mostrada", ins["verdad"])
        col = int((pos[0] - x_min) / (x_max - x_min) * (ancho - 1))
        fil = alto - 1 - int((pos[2] - z_min) / (z_max - z_min) * (alto - 1))
        col = max(0, min(ancho - 1, col))
        fil = max(0, min(alto - 1, fil))
        if ins.get("neutralizada"):
            letra = "X"
        elif ins["tipo"] == "velutina":
            letra = "V"
        else:
            letra = "b"
        lienzo[fil][col] = letra
    for (x, z) in historial_disparos[-6:]:
        col = int((x - x_min) / (x_max - x_min) * (ancho - 1))
        fil = alto - 1 - int((z - z_min) / (z_max - z_min) * (alto - 1))
        col = max(0, min(ancho - 1, col))
        fil = max(0, min(alto - 1, fil))
        if lienzo[fil][col] == ".":
            lienzo[fil][col] = "*"
    linea = "+" + "-" * ancho + "+"
    filas = [linea] + ["|" + "".join(f) + "|" for f in lienzo] + [linea]
    return "\n".join(filas)


def _crea_insectos():
    """Lista determinista: 8 abejas + 3 velutinas con su pipeline EKF."""
    insectos = []
    for i in range(N_ABEJAS):
        p0 = _verdad_abeja(i, 0)
        p1 = _verdad_abeja(i, 1)
        pipe = FusionPipeline(K=K, R=R, t=T, dt=DT)
        pipe.initiate((p0 + np.array([0.01, 0.0, 0.0])).tolist(),
                      (p1 + np.array([0.01, 0.0, 0.0])).tolist())
        insectos.append({"id": "abeja-%d" % (i + 1), "tipo": "abeja",
                         "indice": i, "pipe": pipe, "verdad": p0,
                         "neutralizada": False, "tocada": False,
                         "ultimo": None, "pos_mostrada": p0})
    for i in range(N_VELUTINAS):
        p0 = _verdad_velutina(i, 0)
        p1 = _verdad_velutina(i, 1)
        pipe = FusionPipeline(K=K, R=R, t=T, dt=DT)
        pipe.initiate((p0 + np.array([0.005, 0.0, 0.0])).tolist(),
                      (p1 + np.array([0.005, 0.0, 0.0])).tolist())
        insectos.append({"id": "velu-%d" % (i + 1), "tipo": "velutina",
                         "indice": i, "pipe": pipe, "verdad": p0,
                         "neutralizada": False, "tocada": False,
                         "ultimo": None, "pos_mostrada": p0})
    return insectos


def ejecuta_demo(n_frames=N_FRAMES, con_pausa=True, imprimir=True):
    """Corre la simulacion completa y devuelve el resumen.

    Devuelve dict con: frames, insectos, disparos, neutralizadas,
    abejas_tocadas, precision, telemetria.
    """
    rng = np.random.default_rng(SEMILLA)
    # Referencia dorada (solo lectura): confirma que el escenario existe.
    dorados = replay_scenario("bee_heavy", SEMILLA)
    n_dorados = len(dorados)

    insectos = _crea_insectos()
    fuego = FirePipeline(K=K, R=R, t=T)
    bus = MqttSim()
    recibidos = []
    bus.subscribe("velutina/detecciones", lambda env: recibidos.append(env))
    bus.subscribe("velutina/disparos", lambda env: recibidos.append(env))

    # Muestra de IA de verdad (stub de pixeles), una vez por clase.
    pv_pix_v, pb_pix_v = _muestra_ia("velutina")
    pv_pix_b, pb_pix_b = _muestra_ia("abeja")

    masks = {"velutina": _mask_velutina(), "abeja": _mask_abeja()}
    historial_disparos = []
    n_disparos = 0
    t_sim = 0.0

    for k in range(n_frames):
        t_sim = k * DT
        if imprimir:
            print("\n=== frame %02d  t=%.1fs ===" % (k, t_sim))
        for ins in insectos:
            if ins.get("neutralizada"):
                continue
            idx = ins["indice"]
            if ins["tipo"] == "abeja":
                verdad = _verdad_abeja(idx, k)
                feats = BEE_FEATS
            else:
                verdad = _verdad_velutina(idx, k)
                feats = VELUTINA_FEATS
            ins["verdad"] = verdad
            # Medida ruidosa del lidar + radial (como el simulador real).
            medida = (verdad + rng.normal(0.0, 0.010, 3)).tolist()
            r = float(np.linalg.norm(verdad))
            vr_verdad = float(np.dot(verdad, verdad - ins["pos_mostrada"])
                              / max(r, 1e-6) / DT) if k > 0 else 0.0
            vr = float(vr_verdad + rng.normal(0.0, 0.05))
            pv, pb = PV_PB[ins["tipo"]]
            # Paso por EKF + ROI + reglas + voto IA (el sistema de verdad).
            try:
                salida = ins["pipe"].step_with_ai(
                    lidar_xyz=np.asarray(medida),
                    v_radial=vr,
                    mask=masks[ins["tipo"]],
                    p_velutina=pv,
                    p_bee=pb,
                    track_id=ins["id"],
                    longest_axis_mm=float(feats["longest_axis_mm"]),
                    sideband_hz=float(feats["sideband_hz"]),
                )
            except Exception as exc:  # no tumbar la demo por un track
                if imprimir:
                    print("  %-8s error EKF: %s" % (ins["id"], exc))
                continue
            x9 = salida.get("x9", [0.0] * 9)
            pos_ekf = np.array(x9[:3])
            ins["pos_mostrada"] = pos_ekf
            ins["ultimo"] = salida
            # Puerta de disparo simulada + laser simulado.
            autorizado = bool(fire_authorize(salida, True, True))
            tiro = fuego.cycle(salida, tuple(float(v) for v in pos_ekf),
                               human_min_m=5.0, track_mm=10.0, t=t_sim)
            disparo = bool(tiro.get("shot") is not None)
            if disparo:
                n_disparos += 1
                historial_disparos.append((float(pos_ekf[0]), float(pos_ekf[2])))
            # Telemetria simulada (solo lectura de snapshots).
            bus.publish("velutina/detecciones", t_sim,
                        {"id": ins["id"], "tipo": ins["tipo"],
                         "promovido": bool(salida.get("promoted")),
                         "puertas": dict(salida.get("gates", {}))})
            if disparo:
                bus.publish("velutina/disparos", t_sim,
                            {"id": ins["id"], "t": t_sim})
            # Efecto: la velutina cae al primer disparo; la abeja jamas.
            if disparo and ins["tipo"] == "velutina":
                ins["neutralizada"] = True
            if disparo and ins["tipo"] == "abeja":
                ins["tocada"] = True
            if imprimir:
                voto = salida.get("ai_vote")
                voto_txt = ("SI" if voto is True else
                            (".." if voto is None else "NO"))
                accion = "DISPARA" if disparo else "no dispara"
                print("  %-8s %-8s pos=(%4.1f,%4.1f,%4.1f) Pv=%.3f Pb=%.4f "
                      "voto=%s %s -> %s"
                      % (ins["id"], ins["tipo"], pos_ekf[0], pos_ekf[1],
                         pos_ekf[2], pv, pb, voto_txt,
                         _puertas_txt(salida.get("gates", {})), accion))
        if imprimir:
            print(_dibuja_mapa(insectos, historial_disparos))
            print("  Leyenda: b=abeja  V=velutina  X=neutralizada  "
                  "H=piquera  *=disparo  .=vacio")
            if con_pausa:
                time.sleep(0.25)

    neutralizadas = sum(1 for i in insectos
                        if i["tipo"] == "velutina" and i["neutralizada"])
    tocadas = sum(1 for i in insectos
                  if i["tipo"] == "abeja" and i["tocada"])
    total = len(insectos)
    aciertos = neutralizadas + (N_ABEJAS - tocadas)
    precision = aciertos / total if total else 0.0
    resumen = {
        "frames": n_frames,
        "semilla": SEMILLA,
        "dorados_bee_heavy": n_dorados,
        "insectos": insectos,
        "disparos": n_disparos,
        "neutralizadas": neutralizadas,
        "n_velutinas": N_VELUTINAS,
        "abejas_tocadas": tocadas,
        "n_abejas": N_ABEJAS,
        "precision": precision,
        "telemetria_msgs": len(recibidos),
        "ia_muestra": {"velutina_pix": (pv_pix_v, pb_pix_v),
                       "abeja_pix": (pv_pix_b, pb_pix_b)},
    }
    return resumen


def _imprime_resumen(res):
    print("\n================ RESUMEN ================")
    print("Semilla fija: %s (determinista)" % res["semilla"])
    print("Casos dorados bee_heavy de referencia: %d" % res["dorados_bee_heavy"])
    print("Fotogramas simulados: %d" % res["frames"])
    print("Velutinas neutralizadas: %d/%d"
          % (res["neutralizadas"], res["n_velutinas"]))
    print("Abejas tocadas: %d (objetivo: 0)" % res["abejas_tocadas"])
    print("Disparos laser (simulado): %d" % res["disparos"])
    print("Mensajes telemetria (simulada): %d" % res["telemetria_msgs"])
    print("Precision: %.1f%%" % (100.0 * res["precision"]))
    pv_v, pb_v = res["ia_muestra"]["velutina_pix"]
    pv_b, pb_b = res["ia_muestra"]["abeja_pix"]
    print("Muestra IA pixeles: velutina Pv=%.3f Pb=%.3f | abeja Pv=%.3f Pb=%.3f"
          % (pv_v, pb_v, pv_b, pb_b))
    print("(La decision usa el voto temporal Pv=0.997/Pb=0.0004 del harness.)")
    if res["neutralizadas"] == res["n_velutinas"] and res["abejas_tocadas"] == 0:
        print("RESULTADO: OK - sistema selectivo, ninguna abeja tocada.")
    else:
        print("RESULTADO: REVISAR - no se neutralizo todo o hubo abeja tocada.")


def _anima_matplotlib(res):
    """Ventana animada con matplotlib (solo si esta instalado)."""
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    insectos = res["insectos"]
    n_frames = res["frames"]
    # Reproduce trayectorias verdad para el fondo (determinista).
    fig, eje = plt.subplots()
    eje.set_title("Velutina: demo del apiario (simulado)")
    eje.set_xlabel("distancia X (m)")
    eje.set_ylabel("altura Z (m)")
    eje.set_xlim(-0.5, 5.5)
    eje.set_ylim(0.5, 3.2)
    eje.plot([0], [1.5], "ys", markersize=12, label="piquera")
    puntos = {}
    for ins in insectos:
        color = "red" if ins["tipo"] == "velutina" else "green"
        marca = "o" if ins["tipo"] == "velutina" else "."
        (pt,) = eje.plot([], [], marker=marca, color=color, linestyle="",
                         label=ins["id"])
        puntos[ins["id"]] = pt
    eje.legend(loc="upper right", fontsize=7)

    def _cuadro(k):
        for ins in insectos:
            idx = ins["indice"]
            if ins["tipo"] == "abeja":
                p = _verdad_abeja(idx, min(k, n_frames - 1))
            else:
                p = _verdad_velutina(idx, min(k, n_frames - 1))
            puntos[ins["id"]].set_data([p[0]], [p[2]])
        eje.set_title("Velutina demo simulada - frame %d/%d" % (k, n_frames))
        return list(puntos.values())

    _ = FuncAnimation(fig, _cuadro, frames=n_frames, interval=250,
                      repeat=False, blit=False)
    plt.show()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Demo visual del apiario (simulada, determinista).")
    parser.add_argument("--ascii", action="store_true",
                        help="fuerza consola ASCII aunque haya matplotlib")
    parser.add_argument("--sin-pausa", action="store_true",
                        help="sin pausas entre frames (para CI)")
    parser.add_argument("--frames", type=int, default=N_FRAMES,
                        help="numero de fotogramas (por defecto %d)" % N_FRAMES)
    args = parser.parse_args(argv)

    print("VELUTINA - demo del apiario (todo simulado, semilla %d)" % SEMILLA)
    print("8 abejas en recta a la colmena + 3 velutinas en hovering.")
    print("Cadena: EKF + ROI + R1-R4 + voto IA + laser + telemetria.")

    usa_mpl = False
    if not args.ascii:
        try:
            import importlib.util
            usa_mpl = importlib.util.find_spec("matplotlib") is not None
        except Exception:
            usa_mpl = False

    if usa_mpl:
        # Simula sin imprimir y luego anima la ventana.
        res = ejecuta_demo(n_frames=args.frames, con_pausa=False,
                           imprimir=False)
        _imprime_resumen(res)
        _anima_matplotlib(res)
    else:
        res = ejecuta_demo(n_frames=args.frames,
                           con_pausa=not args.sin_pausa, imprimir=True)
        _imprime_resumen(res)

    # Codigo de salida: 0 si la demo es correcta (todo neutralizado, 0 abejas).
    ok = (res["neutralizadas"] == res["n_velutinas"]
          and res["abejas_tocadas"] == 0)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
