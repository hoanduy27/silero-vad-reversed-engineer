"""Unit tests for silero/trainer/evaluate.py's FAR/FRR math."""
import math

from silero.trainer.evaluate import far_frr


def test_far_frr_perfect_classifier():
    counts = {"tp": 10, "fp": 0, "tn": 20, "fn": 0}
    m = far_frr(counts)
    assert m["far"] == 0.0
    assert m["frr"] == 0.0
    assert m["accuracy"] == 1.0


def test_far_frr_all_wrong():
    counts = {"tp": 0, "fp": 20, "tn": 0, "fn": 10}
    m = far_frr(counts)
    assert m["far"] == 1.0
    assert m["frr"] == 1.0
    assert m["accuracy"] == 0.0


def test_far_is_false_positives_over_actual_negatives():
    # 5 actual negatives (fp=1, tn=4) -> far = 1/5; actual positives untouched
    counts = {"tp": 8, "fp": 1, "tn": 4, "fn": 2}
    m = far_frr(counts)
    assert math.isclose(m["far"], 1 / 5)
    assert math.isclose(m["frr"], 2 / 10)


def test_far_frr_nan_when_no_actual_negatives_or_positives():
    # no actual negatives at all (fp=fn=0 tn=0) -> FAR undefined
    counts = {"tp": 5, "fp": 0, "tn": 0, "fn": 0}
    m = far_frr(counts)
    assert math.isnan(m["far"])
    assert m["frr"] == 0.0

    # no actual positives at all -> FRR undefined
    counts = {"tp": 0, "fp": 0, "tn": 5, "fn": 0}
    m = far_frr(counts)
    assert m["far"] == 0.0
    assert math.isnan(m["frr"])
