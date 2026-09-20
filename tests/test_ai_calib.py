"""RED 2.4c — PR2 calibración T-scaling + punto operativo PR + doubt->bee."""
import math


def test_temperature_scaling_softens_overconfidence():
    from jetson.ai.calibration import scale

    logits = [2.0, 0.5, -1.0]
    p_hot = scale.softmax_with_temperature(logits, 1.0)
    p_soft = scale.softmax_with_temperature(logits, 1.5)
    assert abs(sum(p_hot) - 1.0) < 1e-6
    assert abs(sum(p_soft) - 1.0) < 1e-6
    assert max(p_soft) < max(p_hot), "higher T must soften peak confidence"
    assert p_soft[0] > p_soft[1] > p_soft[2], "order must be preserved"


def test_gate_doubt_defaults_to_bee():
    from jetson.ai.calibration import scale

    # Pv alto pero Pb por encima del máximo → no-fire
    assert scale.gate(0.998, 0.005, r1_r4_ok=True) is False
    # R1-R4 fail → no-fire aunque probs OK
    assert scale.gate(0.999, 0.0005, r1_r4_ok=False) is False
    # todo OK → fire
    assert scale.gate(0.999, 0.0005, r1_r4_ok=True) is True


def test_operating_point_reports_recall_cost():
    from jetson.ai.calibration import scale

    rec = [0.99, 0.995, 1.0]
    prec = [0.98, 0.95, 0.80]
    thr = [0.995, 0.990, 0.900]
    op = scale.find_operating_point(rec, prec, thr, min_recall=0.99)
    assert op["recall"] >= 0.99
    assert "threshold" in op and "precision" in op and "recall_cost" in op
    assert math.isfinite(op["recall_cost"])
