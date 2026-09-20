"""RED 4.1 — PR3 métricas recall/FP/latencia + mAP/PR (falla sin metrics.py)."""


def test_metrics_acceptance_recall_beefp_map_latency():
    from jetson.ai import metrics

    # bee-heavy val sim: 200 velutina (198 TP) + 10000 bees (1 FP)
    y_true = [1] * 200 + [0] * 10000
    y_pred = [1] * 198 + [0] * 2 + [1] * 1 + [0] * 9999
    scores = [0.998] * 198 + [0.50] * 2 + [0.997] * 1 + [0.10] * 9999
    m = metrics.compute(y_true, y_pred, scores, latency_ms=60.0, fps=16.0)
    assert m["recall"] >= 0.99, f"recall {m['recall']} < 0.99"
    assert m["bee_fp_rate"] <= 0.0001, f"bee-FP {m['bee_fp_rate']} > 0.01%"
    assert "map" in m and "pr" in m and "latency_ms" in m and "fps" in m
    assert m["latency_ms"] == 60.0
    # flag 50-70ms CPU como riesgo de transferencia
    assert m["transfer_risk"] is True
    assert "recall_cost" in m


def test_metrics_table_and_pr_point():
    from jetson.ai import metrics

    rec = [0.99, 0.995, 1.0]
    prec = [0.98, 0.95, 0.80]
    thr = [0.995, 0.990, 0.900]
    op = metrics.operating_point(rec, prec, thr, min_recall=0.99)
    assert op["recall"] >= 0.99 and "recall_cost" in op
    t = metrics.as_table({"recall": 0.995, "bee_fp_rate": 0.0001, "map": 0.94,
                          "latency_ms": 60.0, "fps": 16.0, "recall_cost": 0.005})
    assert "recall" in t and "bee_fp" in t.lower() or "bee-fp" in t.lower()
