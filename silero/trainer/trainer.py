"""
Fine-tune Silero VAD's encoder via RepVGG-style BranchedEncoderBlock adapters
(see silero/model_beautified.py), config-driven for adapter-only vs. full
fine-tuning (see egs/vad/webrtc_audit/conf/{adapter_only,full_finetune}.yaml).

Trains in real batches of `train.batch_size` utterances at once: each
utterance is processed chunk-by-chunk (VADRNN.forward_sequence), decoder
LSTMCell state stepped in lockstep across the whole batch, BCE loss against a
per-chunk speech/silence target (silero/data/dataset.py) with padded chunks
masked out. Utterances are duration-bucketed into batches (see
dataset.make_batches) to keep padding waste and the shared per-batch chunk-loop
length close to what each item would need alone.

Every epoch also reports precision/recall/F1 from the epoch's aggregate
confusion matrix (silero/trainer/metrics.py), alongside loss/accuracy.
`train.primary_metric` (default "f1") picks which of these drives both "best
checkpoint" selection and early stopping (train.early_stopping) -- accuracy is
left as a plain observation since it's dominated by whichever class is more
common, which for VAD is usually non-speech.

Usage:
    python -m silero.trainer.trainer --config conf/adapter_only.yaml
"""
import argparse
import copy
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from silero.data.dataset import VadJsonlDataset, collate_batch, load_manifest, make_batches, split_train_val
from silero.model_beautified import VADRNN, BranchedEncoderBlock
from silero.trainer.metrics import add_counts, confusion_counts, precision_recall_f1

# Metrics run_epoch() can report (val_* forms are what train.primary_metric and
# train.early_stopping select from) -> whether higher or lower is better.
METRIC_MODES = {
    "loss": "min",
    "acc": "max",
    "precision": "max",
    "recall": "max",
    "f1": "max",
}


def resolve_repo_path(path: str) -> Path:
    """Resolve a config path against $REPO_ROOT (exported by egs/*/path.sh)
    when it's not found relative to the current working directory -- lets
    yaml configs write repo-relative paths like `assets/silero_vad.jit`
    regardless of which egs/<task>/<dataset>/ directory a script runs from."""
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    repo_root = os.environ.get("REPO_ROOT")
    if repo_root:
        candidate = Path(repo_root) / p
        if candidate.exists():
            return candidate
    return p


def load_base_model(jit_checkpoint: str) -> VADRNN:
    model = VADRNN()
    loaded = torch.jit.load(str(resolve_repo_path(jit_checkpoint)))
    state_dict = loaded._model.state_dict()
    state_dict.pop("stft.forward_basis_buffer", None)
    model.load_state_dict(state_dict, strict=False)
    return model


def branch_encoder(model: VADRNN, use_identity: bool = True) -> VADRNN:
    for i, block in enumerate(model.encoder):
        model.encoder[i] = BranchedEncoderBlock.from_reparam(block, use_identity=use_identity)
    return model


# Names a config's `model.freeze` list can reference -> the module(s) they map to.
FREEZE_TARGETS = {
    "stft": lambda m: [m.stft],
    "decoder": lambda m: [m.decoder],
    "conv3x1": lambda m: [block.conv3x1 for block in m.encoder],
}


def apply_freeze(model: VADRNN, freeze: list):
    for name in freeze:
        if name not in FREEZE_TARGETS:
            raise ValueError(f"Unknown freeze target {name!r} (known: {list(FREEZE_TARGETS)})")
        for module in FREEZE_TARGETS[name](model):
            for p in module.parameters():
                p.requires_grad = False


def build_model(config: dict) -> VADRNN:
    model_cfg = config["model"]
    model = load_base_model(model_cfg["jit_checkpoint"])
    model = branch_encoder(model, use_identity=model_cfg.get("use_identity", True))
    apply_freeze(model, model_cfg.get("freeze", []))
    return model


class RunLogger:
    """Optional TensorBoard/W&B logging, enabled per-config (`logging.tensorboard`,
    `logging.wandb.enabled`). Both are lazily imported so neither is a hard
    dependency of the trainer when left disabled (the default)."""

    def __init__(self, exp_dir: Path, config: dict):
        logging_cfg = config.get("logging", {}) or {}

        self.tb_writer = None
        if logging_cfg.get("tensorboard", False):
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as e:
                raise ImportError(
                    "logging.tensorboard is enabled but the `tensorboard` package "
                    "isn't installed (pip install tensorboard)."
                ) from e
            self.tb_writer = SummaryWriter(log_dir=str(exp_dir / "tensorboard"))

        self.wandb = None
        wandb_cfg = logging_cfg.get("wandb", {}) or {}
        if wandb_cfg.get("enabled", False):
            try:
                import wandb
            except ImportError as e:
                raise ImportError(
                    "logging.wandb.enabled is set but the `wandb` package isn't "
                    "installed (pip install wandb)."
                ) from e
            self.wandb = wandb
            self.wandb.init(
                project=wandb_cfg.get("project", "silero-vad-adapt"),
                name=wandb_cfg.get("run_name") or exp_dir.name,
                config=config,
                dir=str(exp_dir),
            )

    def log_scalars(self, scalars: dict, step: int):
        if self.tb_writer is not None:
            for name, value in scalars.items():
                self.tb_writer.add_scalar(name, value, step)
        if self.wandb is not None:
            self.wandb.log(scalars, step=step)

    def close(self):
        if self.tb_writer is not None:
            self.tb_writer.close()
        if self.wandb is not None:
            self.wandb.finish()


class EarlyStopper:
    """Tracks a metric across epochs; `step()` returns True once `patience`
    consecutive epochs have passed without an improvement of at least
    `min_delta`. mode="min" (e.g. loss) treats lower as better; mode="max"
    (e.g. F1) treats higher as better. Independent of the best-checkpoint
    tracking in main() (which always saves ties/improvements with no
    min_delta) -- this only decides when to stop, not what to save."""

    def __init__(self, patience: int, min_delta: float = 0.0, mode: str = "min"):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float("inf") if mode == "min" else float("-inf")
        self.epochs_since_improvement = 0

    def _improved(self, value: float) -> bool:
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def step(self, value: float) -> bool:
        if self._improved(value):
            self.best = value
            self.epochs_since_improvement = 0
        else:
            self.epochs_since_improvement += 1
        return self.epochs_since_improvement >= self.patience


def average_state_dicts(state_dicts: list) -> dict:
    """Elementwise mean across a list of state_dicts with identical keys/shapes
    (e.g. several epoch checkpoints of the same architecture) -- the standard
    "checkpoint averaging" trick. Note this averages BatchNorm running stats
    directly too rather than recomputing them (no active identity_bn branch on
    the real checkpoint today, see BranchedEncoderBlock, so this doesn't matter
    in practice yet, but would be an approximation if that ever changes)."""
    averaged = {}
    for key in state_dicts[0]:
        stacked = torch.stack([sd[key].float() for sd in state_dicts], dim=0)
        averaged[key] = stacked.mean(dim=0).to(state_dicts[0][key].dtype)
    return averaged


def reparameterize_model(model: VADRNN) -> VADRNN:
    """Deployable copy: every BranchedEncoderBlock folded back into a plain
    SileroVadEncoderBlock (same module shape as the original checkpoint)."""
    fused = copy.deepcopy(model)
    fused.eval()
    for i, block in enumerate(fused.encoder):
        if isinstance(block, BranchedEncoderBlock):
            fused.encoder[i] = block.to_reparam_block()
    return fused


def compute_loss(model: VADRNN, batch: dict, device):
    """Returns None if the batch has no valid (non-padded) frames, else a dict
    with `loss` (tensor, for backward), `n` (valid frame count, for weighting
    across batches), and `counts` (this batch's confusion counts, for
    aggregate-epoch precision/recall/F1 -- see silero/trainer/metrics.py)."""
    audio = batch["audio"].to(device)
    target = batch["target"].to(device)
    valid_frames = batch["valid_frames"].to(device)

    probs = model.forward_sequence(audio)  # (B, frames)
    n_frames = min(probs.size(1), target.size(1))
    probs, target = probs[:, :n_frames], target[:, :n_frames]

    # mask[b, t] = t < valid_frames[b] -- excludes padded (t >= that item's
    # true length) frames from the loss/metrics entirely.
    frame_idx = torch.arange(n_frames, device=device).unsqueeze(0)
    mask = frame_idx < valid_frames.clamp(max=n_frames).unsqueeze(1)

    probs_valid = probs[mask]
    target_valid = target[mask]
    n = probs_valid.numel()
    if n == 0:
        return None

    loss = F.binary_cross_entropy(probs_valid, target_valid)
    preds_valid = probs_valid.detach() >= 0.5
    counts = confusion_counts(preds_valid, target_valid)
    return {"loss": loss, "n": n, "counts": counts}


def run_epoch(model: VADRNN, rows: list, manifest_path: Path, device,
              optimizer=None, batch_size: int = 1, shuffle: bool = False, seed: int = 0,
              max_frames=None) -> dict:
    """Returns {"loss", "acc", "precision", "recall", "f1"} aggregated over the
    whole epoch (precision/recall/F1 from the epoch's aggregate confusion
    matrix, not averaged per-batch -- they're non-linear, so that would be
    wrong). With speech frames a small minority of this dataset's total, `acc`
    is dominated by the majority (non-speech) class and mainly useful as a
    sanity-check observation, not a decision signal -- see train.primary_metric."""
    training = optimizer is not None
    # max_frames only matters for training (full BPTT through the whole
    # sequence retains per-step activations for backward); eval runs
    # forward-only under torch.no_grad() below, so it's left uncapped.
    dataset = VadJsonlDataset(manifest_path, rows=rows, max_frames=max_frames if training else None)
    model.train(training)

    batches = make_batches(rows, batch_size, shuffle=shuffle, seed=seed)

    total_loss, total_frames = 0.0, 0
    total_counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

    with torch.enable_grad() if training else torch.no_grad():
        for batch_indices in batches:
            items = [dataset[i] for i in batch_indices]
            items = [it for it in items if it["target"].numel() > 0]
            if not items:
                continue

            batch = collate_batch(items)
            result = compute_loss(model, batch, device)
            if result is None:
                continue
            loss, n, counts = result["loss"], result["n"], result["counts"]

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * n
            total_frames += n
            total_counts = add_counts(total_counts, counts)

    if total_frames == 0:
        return {"loss": float("nan"), "acc": float("nan"),
                "precision": float("nan"), "recall": float("nan"), "f1": float("nan")}

    acc = (total_counts["tp"] + total_counts["tn"]) / total_frames
    return {"loss": total_loss / total_frames, "acc": acc, **precision_recall_f1(total_counts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--exp-dir", type=Path, default=None,
                         help="Overrides the config's exp_dir")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    exp_dir = args.exp_dir or Path(config["exp_dir"])
    exp_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = config["train"]
    device_str = train_cfg.get("device", "cpu")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"config requests device={device_str!r} but torch.cuda.is_available() is False "
            "-- set train.device to 'cpu' if no GPU is available here."
        )
    device = torch.device(device_str)
    seed = train_cfg.get("seed", 0)
    torch.manual_seed(seed)

    manifest_path = resolve_repo_path(config["data"]["manifest"])
    rows = load_manifest(manifest_path)
    train_rows, val_rows = split_train_val(rows, config["data"].get("val_fraction", 0.2), seed=seed)
    print(f"train: {len(train_rows)} files, val: {len(val_rows)} files")

    model = build_model(config).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {n_trainable}/{n_total} ({100 * n_trainable / n_total:.2f}%)")

    optimizer = torch.optim.AdamW(trainable_params, lr=float(train_cfg["lr"]))
    batch_size = max(1, train_cfg.get("batch_size", 1))
    max_utterance_seconds = train_cfg.get("max_utterance_seconds")
    max_frames = round(max_utterance_seconds * 16000 / 512) if max_utterance_seconds else None
    logger = RunLogger(exp_dir, config)

    # Drives BOTH "best checkpoint" selection and early stopping (if enabled) --
    # keeping them on the same metric avoids the confusing case of training
    # stopping because one metric plateaued while "best.pt" reflects a
    # different epoch chosen by another. Default f1: with speech frames a
    # small minority of this dataset's total, accuracy alone is misleading.
    primary_metric = train_cfg.get("primary_metric", "f1")
    if primary_metric not in METRIC_MODES:
        raise ValueError(f"Unknown train.primary_metric {primary_metric!r} (known: {list(METRIC_MODES)})")
    metric_mode = METRIC_MODES[primary_metric]

    def is_better(new_value: float, best_value: float) -> bool:
        return new_value >= best_value if metric_mode == "max" else new_value <= best_value

    early_stop_cfg = train_cfg.get("early_stopping", {}) or {}
    early_stopper = (
        EarlyStopper(patience=early_stop_cfg["patience"], min_delta=early_stop_cfg.get("min_delta", 0.0), mode=metric_mode)
        if early_stop_cfg.get("enabled", False) else None
    )

    avg_cfg = train_cfg.get("model_averaging", {}) or {}
    averaging_enabled = avg_cfg.get("enabled", False)
    checkpoints_dir = exp_dir / "checkpoints"
    if averaging_enabled:
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_metric_value = float("-inf") if metric_mode == "max" else float("inf")
    epochs_run = 0
    try:
        for epoch in range(train_cfg["epochs"]):
            train_metrics = run_epoch(
                model, train_rows, manifest_path, device, optimizer,
                batch_size=batch_size, shuffle=True, seed=seed + epoch, max_frames=max_frames,
            )
            val_metrics = (
                run_epoch(model, val_rows, manifest_path, device, batch_size=batch_size)
                if val_rows else {"loss": float("nan"), "acc": float("nan"),
                                   "precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
            )
            epochs_run = epoch + 1
            current_value = val_metrics[primary_metric]

            print(
                f"epoch {epoch}: "
                f"train[loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.4f} f1={train_metrics['f1']:.4f}] "
                f"val[loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.4f} "
                f"precision={val_metrics['precision']:.4f} recall={val_metrics['recall']:.4f} f1={val_metrics['f1']:.4f}]"
            )
            history.append({
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            })
            logger.log_scalars({
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **{f"val/{k}": v for k, v in val_metrics.items()},
            }, step=epoch)

            torch.save(model.state_dict(), exp_dir / "last.pt")
            if is_better(current_value, best_metric_value):
                best_metric_value = current_value
                torch.save(model.state_dict(), exp_dir / "best.pt")
                torch.save(reparameterize_model(model).state_dict(), exp_dir / "best_fused.pt")
            if averaging_enabled:
                torch.save(model.state_dict(), checkpoints_dir / f"epoch_{epoch:03d}.pt")

            if early_stopper is not None and early_stopper.step(current_value):
                print(
                    f"early stopping at epoch {epoch} "
                    f"(no val_{primary_metric} improvement for {early_stop_cfg['patience']} epochs)"
                )
                break
    finally:
        logger.close()

    with open(exp_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    if averaging_enabled:
        strategy = avg_cfg.get("strategy", "last_n")
        n = min(avg_cfg.get("n", 5), epochs_run)
        if strategy == "last_n":
            selected_epochs = list(range(epochs_run - n, epochs_run))
        elif strategy == "best_k":
            selected_epochs = sorted(
                range(epochs_run), key=lambda e: history[e][f"val_{primary_metric}"], reverse=(metric_mode == "max"),
            )[:n]
        else:
            raise ValueError(f"Unknown model_averaging.strategy: {strategy!r} (expected 'last_n' or 'best_k')")

        print(f"model averaging: strategy={strategy}, averaging epochs {selected_epochs}")
        state_dicts = [torch.load(checkpoints_dir / f"epoch_{e:03d}.pt", map_location="cpu") for e in selected_epochs]
        averaged_state = average_state_dicts(state_dicts)

        averaged_model = build_model(config)
        averaged_model.load_state_dict(averaged_state)
        averaged_model.eval()
        torch.save(averaged_model.state_dict(), exp_dir / "averaged.pt")
        torch.save(reparameterize_model(averaged_model).state_dict(), exp_dir / "averaged_fused.pt")
        print(f"saved -> {exp_dir / 'averaged.pt'}, {exp_dir / 'averaged_fused.pt'}")


if __name__ == "__main__":
    main()
