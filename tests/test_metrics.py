"""Unit tests for silero/trainer/metrics.py."""
import math

import torch

from silero.trainer.metrics import add_counts, confusion_counts, far_frr, precision_recall_f1


def test_confusion_counts_basic():
    preds = torch.tensor([1, 1, 0, 0, 1])
    target = torch.tensor([1, 0, 0, 1, 1])
    counts = confusion_counts(preds, target)
    assert counts == {"tp": 2, "fp": 1, "tn": 1, "fn": 1}


def test_confusion_counts_accepts_bool_tensors():
    preds = torch.tensor([True, False])
    target = torch.tensor([True, True])
    counts = confusion_counts(preds, target)
    assert counts == {"tp": 1, "fp": 0, "tn": 0, "fn": 1}


def test_add_counts_sums_matching_keys():
    a = {"tp": 1, "fp": 2, "tn": 3, "fn": 4}
    b = {"tp": 10, "fp": 20, "tn": 30, "fn": 40}
    assert add_counts(a, b) == {"tp": 11, "fp": 22, "tn": 33, "fn": 44}


def test_add_counts_missing_keys_default_to_zero():
    assert add_counts({}, {"tp": 5}) == {"tp": 5, "fp": 0, "tn": 0, "fn": 0}


def test_precision_recall_f1_perfect_classifier():
    counts = {"tp": 10, "fp": 0, "tn": 20, "fn": 0}
    m = precision_recall_f1(counts)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0


def test_precision_recall_f1_known_values():
    # precision = 5/(5+5) = 0.5, recall = 5/(5+15) = 0.25
    # f1 = 2*0.5*0.25/(0.5+0.25) = 0.25/0.75 = 1/3
    counts = {"tp": 5, "fp": 5, "tn": 0, "fn": 15}
    m = precision_recall_f1(counts)
    assert math.isclose(m["precision"], 0.5)
    assert math.isclose(m["recall"], 0.25)
    assert math.isclose(m["f1"], 1 / 3)


def test_precision_recall_f1_no_predicted_positives_is_nan_precision_zero_recall():
    counts = {"tp": 0, "fp": 0, "tn": 10, "fn": 5}
    m = precision_recall_f1(counts)
    assert math.isnan(m["precision"])  # no positive predictions at all
    assert m["recall"] == 0.0
    assert math.isnan(m["f1"])


def test_precision_recall_f1_no_actual_positives_is_nan_recall():
    counts = {"tp": 0, "fp": 5, "tn": 10, "fn": 0}
    m = precision_recall_f1(counts)
    assert m["precision"] == 0.0
    assert math.isnan(m["recall"])
    assert math.isnan(m["f1"])


def test_f1_harmonic_mean_penalizes_imbalance_more_than_arithmetic_mean_would():
    # precision=1.0, recall=0.01 -- arithmetic mean would be ~0.5, harmonic mean
    # (F1) should be much lower, reflecting how badly recall drags it down.
    counts = {"tp": 1, "fp": 0, "tn": 0, "fn": 99}
    m = precision_recall_f1(counts)
    assert m["precision"] == 1.0
    assert math.isclose(m["recall"], 0.01)
    assert m["f1"] < 0.05


def test_far_frr_matches_precision_recall_relationship():
    counts = {"tp": 8, "fp": 1, "tn": 4, "fn": 2}
    frr = far_frr(counts)
    pr = precision_recall_f1(counts)
    # recall == 1 - FRR by definition (both are about actual positives)
    assert math.isclose(pr["recall"], 1 - frr["frr"])
