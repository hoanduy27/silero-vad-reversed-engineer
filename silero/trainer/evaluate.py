"""
Evaluate a fine-tuned VAD checkpoint's FAR/FRR/precision/recall/F1 on the
held-out validation split (the same stratified split trainer.py used,
recomputed from the same manifest/seed -- not persisted separately), alongside
the original, un-adapted checkpoint as a baseline for comparison.

FAR (False Acceptance Rate): fraction of true non-speech frames the model
    calls speech (false positives among actual negatives).
FRR (False Rejection Rate): fraction of true speech frames the model calls
    non-speech (false negatives among actual positives).
Precision/recall/F1: see silero/trainer/metrics.py -- with speech frames a
    small minority of the total (this dataset's ~9%), these (not accuracy)
    are what actually reflect speech-detection quality.

Usage:
    python -m silero.trainer.evaluate --config conf/adapter_only.yaml
"""
import argparse
from pathlib import Path

import torch
import yaml

from silero.data.dataset import VadJsonlDataset, load_manifest, split_train_val
from silero.model_beautified import VADRNN
from silero.trainer.metrics import add_counts, far_frr, precision_recall_f1
from silero.trainer.metrics import confusion_counts as frame_confusion_counts
from silero.trainer.trainer import load_base_model, resolve_repo_path


def confusion_counts(model: VADRNN, rows: list, manifest_path: Path, device, threshold: float = 0.5) -> dict:
    """Aggregate confusion counts over every row's valid frames."""
    dataset = VadJsonlDataset(manifest_path, rows=rows)
    model.eval()

    total = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    with torch.no_grad():
        for i in range(len(dataset)):
            item = dataset[i]
            if item["target"].numel() == 0:
                continue
            audio = item["audio"].to(device)
            target = item["target"].to(device)

            probs = model.forward_sequence(audio).squeeze(0)
            n = min(probs.size(0), target.size(0))
            preds = probs[:n] >= threshold
            total = add_counts(total, frame_confusion_counts(preds, target[:n]))

    return total


def load_fused_checkpoint(checkpoint_path: Path) -> VADRNN:
    """Load a reparameterized checkpoint (e.g. exp/<variant>/best_fused.pt) --
    same plain SileroVadEncoderBlock module shape as the original .jit checkpoint,
    so this is a straight state_dict load, no key surgery needed."""
    model = VADRNN()
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model


def format_report(exp_name: str, n_val_files: int, threshold: float,
                   baseline: dict, finetuned: dict, checkpoint_name: str) -> str:
    def row(label, m):
        return (
            f"| {label} | {m['far']:.4f} | {m['frr']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} "
            f"| {m['f1']:.4f} | {m['accuracy']:.4f} | {m['tp']} | {m['fp']} | {m['tn']} | {m['fn']} |"
        )

    n_actual_positive = finetuned["tp"] + finetuned["fn"]
    n_total = n_actual_positive + finetuned["fp"] + finetuned["tn"]
    positive_rate = n_actual_positive / n_total if n_total > 0 else float("nan")

    lines = [
        f"# FAR/FRR report ({exp_name})",
        "",
        f"Held-out val: {n_val_files} files, decision threshold={threshold}, "
        f"{100 * positive_rate:.1f}% of frames are actual speech",
        "",
        "FAR = false positives / actual negatives (non-speech misflagged as speech).",
        "FRR = false negatives / actual positives (speech misflagged as non-speech).",
        "Precision/recall/F1 (positive class = speech) -- with speech this much a "
        "minority of frames, these (not accuracy) reflect detection quality.",
        "",
        "| model | FAR | FRR | precision | recall | F1 | accuracy | TP | FP | TN | FN |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
        row("baseline (original checkpoint)", baseline),
        row(f"fine-tuned ({checkpoint_name})", finetuned),
    ]
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None,
                         help="Fused checkpoint to evaluate (default: <exp_dir>/best_fused.pt)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=None,
                         help="Report markdown path (default: <exp_dir>/far_frr_report.md)")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    exp_dir = Path(config["exp_dir"])
    checkpoint_path = args.checkpoint or (exp_dir / "best_fused.pt")
    out_path = args.out or (exp_dir / "far_frr_report.md")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"{checkpoint_path} not found -- run training first (local/train.sh)")

    device = torch.device("cpu")  # one-shot eval; not worth the config's train device

    manifest_path = resolve_repo_path(config["data"]["manifest"])
    rows = load_manifest(manifest_path)
    _, val_rows = split_train_val(rows, config["data"].get("val_fraction", 0.2), seed=config["train"].get("seed", 0))
    print(f"evaluating on {len(val_rows)} held-out val files (threshold={args.threshold})")

    baseline_model = load_base_model(config["model"]["jit_checkpoint"]).to(device)
    finetuned_model = load_fused_checkpoint(checkpoint_path).to(device)

    baseline_counts = confusion_counts(baseline_model, val_rows, manifest_path, device, args.threshold)
    finetuned_counts = confusion_counts(finetuned_model, val_rows, manifest_path, device, args.threshold)
    baseline_metrics = {**far_frr(baseline_counts), **precision_recall_f1(baseline_counts)}
    finetuned_metrics = {**far_frr(finetuned_counts), **precision_recall_f1(finetuned_counts)}

    report = format_report(
        exp_dir.name, len(val_rows), args.threshold,
        baseline_metrics, finetuned_metrics, checkpoint_path.name,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report)

    print(report)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
