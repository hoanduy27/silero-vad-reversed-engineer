"""
Confusion-matrix-derived metrics shared by training (silero/trainer/trainer.py)
and evaluation (silero/trainer/evaluate.py), so both compute FAR/FRR/precision/
recall/F1 the same way.

With this dataset's speech frames a small minority of total frames (~9% in the
current webrtc_audit split), accuracy is dominated by the majority (non-speech)
class and can look deceptively high while the model misses most real speech --
precision/recall/F1 (computed from the *aggregate* confusion matrix over an
epoch or eval set, not averaged per-batch, since F1 is non-linear) are the
metrics that actually reflect speech-detection quality here.
"""
import math

import torch

CONFUSION_KEYS = ("tp", "fp", "tn", "fn")


def confusion_counts(preds: torch.Tensor, target: torch.Tensor) -> dict:
    """preds, target: same-shape tensors of 0/1 (or bool) values."""
    preds = preds.bool()
    target = target.bool()
    return {
        "tp": int((preds & target).sum().item()),
        "fp": int((preds & ~target).sum().item()),
        "tn": int((~preds & ~target).sum().item()),
        "fn": int((~preds & target).sum().item()),
    }


def add_counts(a: dict, b: dict) -> dict:
    return {k: a.get(k, 0) + b.get(k, 0) for k in CONFUSION_KEYS}


def far_frr(counts: dict) -> dict:
    """FAR = false positives / actual negatives. FRR = false negatives / actual positives."""
    n_actual_negative = counts["fp"] + counts["tn"]
    n_actual_positive = counts["fn"] + counts["tp"]
    n_total = n_actual_negative + n_actual_positive

    far = counts["fp"] / n_actual_negative if n_actual_negative > 0 else float("nan")
    frr = counts["fn"] / n_actual_positive if n_actual_positive > 0 else float("nan")
    accuracy = (counts["tp"] + counts["tn"]) / n_total if n_total > 0 else float("nan")

    return {"far": far, "frr": frr, "accuracy": accuracy, **counts}


def precision_recall_f1(counts: dict) -> dict:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

    f1 = float("nan")
    if not (math.isnan(precision) or math.isnan(recall)) and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}
