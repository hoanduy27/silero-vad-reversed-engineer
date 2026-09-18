"""
Test the RepVGG-style structural re-parameterization hypothesis for the
`reparam_conv` layers in Silero's encoder.

Hypothesis: each `reparam_conv` (a plain Conv1d(k=3) at inference) was actually
trained as a multi-branch block (e.g. conv3x1 + conv1x1 [+ identity], each with
its own BatchNorm) and later algebraically fused into a single Conv1d for
deployment -- the same trick RepVGG uses for CNNs.

If that's true, the *fused* kernel should carry a structural fingerprint:
- A 1x1 branch only ever contributes to the *center* tap of a fused 3x1 kernel
  (padding a 1x1 kernel to width 3 puts its single value in the middle), so
  blocks with a 1x1 branch should show a center tap that is disproportionately
  large/energetic compared to the two edge taps, which come purely from the
  3x1 branch.
- An identity branch (only possible when in_channels == out_channels and
  stride == 1) additionally adds a scaled identity matrix into that same
  center tap, so the center tap's diagonal should dominate its off-diagonal
  entries for blocks where that's structurally possible.

This script measures both signatures directly on the real checkpoint weights.
It can't *prove* the exact training-time branch topology (fusion is a lossy,
non-invertible sum), but a clear center-tap spike is strong circumstantial
evidence for the RepVGG hypothesis, and its absence is evidence against it.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def tap_stats(weight: torch.Tensor):
    """weight: (out_channels, in_channels, kernel_size). Returns per-tap mean|w| and std."""
    mean_abs = weight.abs().mean(dim=(0, 1))
    std = weight.std(dim=(0, 1))
    return mean_abs, std


def diagonal_signature(weight: torch.Tensor, tap: int):
    """For square (out==in) blocks, compare the center tap's diagonal vs off-diagonal energy."""
    center = weight[:, :, tap]
    if center.shape[0] != center.shape[1]:
        return None
    diag = torch.diagonal(center)
    off = center - torch.diag_embed(diag)
    off_vals = off[off.abs() > 0]
    return {
        "diag_abs_mean": diag.abs().mean().item(),
        "offdiag_abs_mean": off_vals.abs().mean().item(),
    }


def analyze_block(name: str, weight: torch.Tensor, bias: torch.Tensor):
    out_c, in_c, k = weight.shape
    center_tap = k // 2

    mean_abs, std = tap_stats(weight)
    edge_mean_abs = torch.cat([mean_abs[:center_tap], mean_abs[center_tap + 1:]]).mean().item()
    edge_std = torch.cat([std[:center_tap], std[center_tap + 1:]]).mean().item()
    center_mean_abs = mean_abs[center_tap].item()
    center_std = std[center_tap].item()

    ratio_mean = center_mean_abs / edge_mean_abs if edge_mean_abs > 0 else float("inf")
    ratio_std = center_std / edge_std if edge_std > 0 else float("inf")

    print(f"[{name}] shape={tuple(weight.shape)}")
    print(f"  mean|w| per tap: {[round(v, 4) for v in mean_abs.tolist()]}")
    print(f"  std   w  per tap: {[round(v, 4) for v in std.tolist()]}")
    print(f"  bias mean/std: {bias.mean().item():.4f} / {bias.std().item():.4f}")
    print(f"  center/edge ratio -- mean|w|: {ratio_mean:.2f}x, std: {ratio_std:.2f}x")

    diag_sig = diagonal_signature(weight, center_tap)
    if diag_sig is not None:
        print(
            f"  [square block] center-tap diagonal vs off-diagonal |w|: "
            f"{diag_sig['diag_abs_mean']:.4f} vs {diag_sig['offdiag_abs_mean']:.4f}"
        )

    verdict_1x1 = ratio_mean > 3.0 or ratio_std > 3.0
    verdict_identity = diag_sig is not None and diag_sig["diag_abs_mean"] > 1.5 * diag_sig["offdiag_abs_mean"]
    print(f"  -> center-tap (1x1-branch) signature: {'YES' if verdict_1x1 else 'no'}")
    if diag_sig is not None:
        print(f"  -> identity-branch signature: {'YES' if verdict_identity else 'no'}")
    print()

    return verdict_1x1, verdict_identity


def plot_deviation_heatmaps(sd, num_blocks: int = 4, out_path: str = "log/rep_vgg_heatmap.png"):
    """
    Per-(out_channel, in_channel) visualization of the center-tap signal, since
    the scalar mean|w|/std ratios in analyze_block() collapse away the spatial
    structure that actually distinguishes "dense fused 1x1 branch" (a full,
    textured out x in grid) from "a few outlier channels" (a sparse, heavy-tailed
    grid with a handful of extreme pixels).

    Top row:    deviation[o, i] = center_tap[o, i] - mean(edge_taps[o, i])
                (a dashed diagonal is overlaid for square out==in blocks, since
                that's where an identity branch's contribution would land)
    Bottom row: raw center tap weight, for reference

    Also prints how concentrated the deviation is (top-1% pixel share, max/median
    ratio) -- a highly concentrated/heavy-tailed map argues against a dense
    fused-branch explanation even when the aggregate ratio looks large.
    """
    fig, axes = plt.subplots(2, num_blocks, figsize=(5 * num_blocks, 9))

    for i in range(num_blocks):
        w = sd[f"encoder.{i}.reparam_conv.weight"]  # (out, in, k)
        out_c, in_c, k = w.shape
        center = k // 2

        edge_taps = torch.cat([w[:, :, :center], w[:, :, center + 1:]], dim=2)
        edge_mean = edge_taps.mean(dim=2)
        center_tap = w[:, :, center]
        deviation = (center_tap - edge_mean).numpy()

        abs_dev = np.abs(deviation).flatten()
        total = abs_dev.sum()
        top1pct_n = max(1, int(len(abs_dev) * 0.01))
        top1pct_share = np.sort(abs_dev)[-top1pct_n:].sum() / total if total > 0 else 0.0
        median = np.median(abs_dev)
        max_dev = abs_dev.max()
        print(
            f"[encoder.{i}] deviation concentration: top1% of pixels hold "
            f"{100 * top1pct_share:.1f}% of total |deviation|, "
            f"max/median = {max_dev / median if median > 0 else float('inf'):.1f}x"
        )

        vmax = np.abs(deviation).max()
        ax = axes[0, i]
        im = ax.imshow(deviation, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(f"encoder.{i}: center - mean(edges)\nshape={tuple(w.shape)}")
        ax.set_xlabel("in_channel")
        ax.set_ylabel("out_channel")
        if out_c == in_c:
            ax.plot(range(in_c), range(out_c), color="lime", lw=0.6, alpha=0.5, ls="--")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax2 = axes[1, i]
        im2 = ax2.imshow(center_tap.numpy(), cmap="viridis", aspect="auto")
        ax2.set_title(f"encoder.{i}: raw center tap")
        ax2.set_xlabel("in_channel")
        ax2.set_ylabel("out_channel")
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"\nSaved heatmap to {out_path}")


def main():
    loaded = torch.jit.load("assets/silero_vad.jit")
    sd = loaded._model.state_dict()

    print("=" * 70)
    print("RepVGG structural re-parameterization hypothesis check")
    print("=" * 70)
    print()

    results = []
    for i in range(4):
        w = sd[f"encoder.{i}.reparam_conv.weight"]
        b = sd[f"encoder.{i}.reparam_conv.bias"]
        results.append(analyze_block(f"encoder.{i}", w, b))

    n_1x1 = sum(r[0] for r in results)
    n_identity = sum(1 for r in results if r[1])
    print("-" * 70)
    print(f"Summary: {n_1x1}/4 blocks show a center-tap (1x1-branch) signature.")
    print(f"Summary: {n_identity}/4 square blocks show an identity-branch signature.")
    print(
        "Interpretation: a center-tap spike with no matching diagonal dominance "
        "is consistent with fused conv3x1 + conv1x1 branches (no identity branch) "
        "-- e.g. because those blocks change channel count and/or use stride=2, "
        "both of which rule out a RepVGG identity branch."
    )

    print()
    print("-" * 70)
    print("Per-channel-pair deviation heatmaps")
    print("-" * 70)
    plot_deviation_heatmaps(sd)


if __name__ == "__main__":
    main()
