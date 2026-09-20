"""Humo del pack de evaluación (rápido, <60 s, sin ventanas).

Verifica que los 4 scripts nuevos funcionan:
- eval_by_condition (solo bee_heavy para ir rápido)
- monte_carlo con 3 semillas
- sweep con 2 valores
- stress básico (las 4 pruebas)
"""

from sim.eval_by_condition import evalua
from sim.monte_carlo import monte_carlo
from sim.stress import pruebas_estres
from sim.sweep import barrido


def test_eval_bee_heavy_humo():
    res = evalua(condiciones=["bee_heavy"])
    f = res["por_condicion"]["bee_heavy"]
    assert f["n"] == 100
    assert 0.0 <= f["recall"] <= 1.0
    assert f["fp_abeja"] == 0.0, "duda->abeja: FP debe ser 0"
    assert f["precision"] >= 0.95
    assert 0.0 <= f["iou"] <= 1.0
    cm = res["matriz"]
    assert cm["TP"] + cm["FN"] + cm["FP"] + cm["TN"] == 100


def test_monte_carlo_humo_3_semillas():
    res = monte_carlo(n_semillas=3, semilla_base=7)
    assert res["n"] == 3
    for nombre in ("recall", "fp", "precision"):
        f = res[nombre]
        assert 0.0 <= f["media"] <= 1.0
        assert f["ic_inf"] <= f["media"] <= f["ic_sup"]
    assert res["fp"]["media"] == 0.0


def test_sweep_humo_2_valores():
    filas = barrido("distancia", [2.0, 4.0], semilla=7)
    assert len(filas) == 2
    for f in filas:
        assert f["n"] == 100
        assert 0.0 <= f["recall"] <= 1.0
        assert f["fp"] == 0.0


def test_stress_humo():
    res = pruebas_estres()
    assert len(res["pruebas"]) == 4
    nombres = [p["nombre"] for p in res["pruebas"]]
    assert any("enjambre" in n for n in nombres)
    assert any("LiDAR" in n for n in nombres)
    assert any("latencia" in n for n in nombres)
    assert any("shutter" in n for n in nombres)
    for p in res["pruebas"]:
        assert "criterio" in p and "detalle" in p
        assert isinstance(p["pass"], bool)
    assert res["ok_total"] is True, "todas las pruebas de rotura deben pasar"
