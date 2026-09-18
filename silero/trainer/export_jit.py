"""
Export a fine-tuned checkpoint back to TorchScript, matching the structure
Silero itself ships (e.g. assets/silero_vad.jit): a scripted VADRNNJITMerge-
style wrapper with a `_model` (16kHz) and `_model_8k` (8kHz) submodule, each
exposing forward(x, state) -> (out, new_state), loadable and usable exactly
like `torch.jit.load(...)` already is everywhere else in this repo.

`_model` is our fine-tuned, fused (single conv3x1, no branches) checkpoint.
`_model_8k` is reused as-is from the original checkpoint -- this pipeline
never touches the 8kHz model, so there's nothing to fine-tune there; it's
carried over only so the export is a structurally complete drop-in
replacement, not because its weights changed.

Checks two things should be numerically identical (fusion and scripting are
both supposed to be lossless), through the same VAD.forward(x, sr) streaming
API real deployment uses (not just the inner VADRNN.forward(x, state)):
  1. the trained BRANCHED model (best.pt) vs. its reparameterized FUSED eager
     form (best_fused.pt), both wrapped in VAD, at 16kHz
  2. that fused form vs. the same wrapper after torch.jit.script + save +
     reload, at 16kHz AND 8kHz (the latter exercises the untouched
     `_model_8k` passthrough)

Usage:
    python -m silero.trainer.export_jit --config conf/adapter_only.yaml
"""
import argparse
from pathlib import Path

import torch
import yaml

from silero.model_beautified import VAD, VADRNN
from silero.trainer.evaluate import load_fused_checkpoint
from silero.trainer.trainer import build_model, resolve_repo_path


def _run_streaming(vad: VAD, chunks: list, sr: int) -> list:
    """Feed `chunks` through vad.forward(x, sr) one at a time -- the same
    streaming API (context/state carried internally) real deployment uses."""
    vad.reset_states()
    outs = []
    with torch.no_grad():
        for chunk in chunks:
            outs.append(vad(chunk, sr=sr))
    return outs


def make_vad(model: VADRNN, model_8k) -> VAD:
    vad = VAD()
    vad._model = model
    vad._model_8k = model_8k
    vad.eval()
    return vad


def check_numerical_parity(branched_vad: VAD, fused_vad: VAD, reloaded_vad) -> float:
    torch.manual_seed(0)
    chunks_16k = [torch.rand(1, 512) for _ in range(20)]
    chunks_8k = [torch.rand(1, 256) for _ in range(20)]

    outs_branched = _run_streaming(branched_vad, chunks_16k, sr=16000)
    outs_fused = _run_streaming(fused_vad, chunks_16k, sr=16000)
    outs_reloaded_16k = _run_streaming(reloaded_vad, chunks_16k, sr=16000)
    outs_reloaded_8k = _run_streaming(reloaded_vad, chunks_8k, sr=8000)

    branch_vs_fused = max((a - b).abs().max().item() for a, b in zip(outs_branched, outs_fused))
    fused_vs_scripted = max((a - b).abs().max().item() for a, b in zip(outs_fused, outs_reloaded_16k))

    print(f"branched vs fused (reparameterize), 16kHz:        max abs diff = {branch_vs_fused:.3e}")
    print(f"fused vs scripted+reloaded (torch.jit), 16kHz:     max abs diff = {fused_vs_scripted:.3e}")
    print(f"scripted+reloaded 8kHz passthrough ran OK ({len(outs_reloaded_8k)} chunks, _model_8k untouched)")
    return max(branch_vs_fused, fused_vs_scripted)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fused-checkpoint", type=Path, default=None,
                         help="Default: <exp_dir>/best_fused.pt")
    parser.add_argument("--branched-checkpoint", type=Path, default=None,
                         help="Default: <exp_dir>/best.pt")
    parser.add_argument("--out", type=Path, default=None,
                         help="Default: <exp_dir>/model.jit")
    parser.add_argument("--tolerance", type=float, default=1e-4,
                         help="Max acceptable abs diff before treating parity as a failure")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    exp_dir = Path(config["exp_dir"])
    fused_checkpoint = args.fused_checkpoint or (exp_dir / "best_fused.pt")
    branched_checkpoint = args.branched_checkpoint or (exp_dir / "best.pt")
    out_path = args.out or (exp_dir / "model.jit")

    for p in (fused_checkpoint, branched_checkpoint):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found -- run training first (local/train.sh)")

    original = torch.jit.load(str(resolve_repo_path(config["model"]["jit_checkpoint"])))
    model_8k = original._model_8k

    branched_model = build_model(config)
    branched_model.load_state_dict(torch.load(branched_checkpoint, map_location="cpu"))
    branched_model.eval()

    fused_model = load_fused_checkpoint(fused_checkpoint)
    fused_model.eval()

    branched_vad = make_vad(branched_model, model_8k)
    fused_vad = make_vad(fused_model, model_8k)

    scripted = torch.jit.script(fused_vad)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(out_path))
    print(f"exported -> {out_path}")

    reloaded_vad = torch.jit.load(str(out_path))
    max_diff = check_numerical_parity(branched_vad, fused_vad, reloaded_vad)

    if max_diff > args.tolerance:
        raise RuntimeError(
            f"numerical parity check FAILED: max abs diff {max_diff:.3e} > tolerance {args.tolerance:.3e}"
        )
    print(f"numerical parity OK (max abs diff {max_diff:.3e} <= tolerance {args.tolerance:.3e})")


if __name__ == "__main__":
    main()
