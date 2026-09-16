"""
Project speaker embeddings (from compute_embeddings.py) to 2D with t-SNE and
plot them, colored by speaker.

Usage:
    python -m silero.speaker_sim.plot_tsne --level utterance \
        --embeddings exp/embeddings/embeddings.npz --out exp/plots/tsne_speaker_embeddings.png
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_EMBEDDINGS = {
    "utterance": SCRIPT_DIR / "data" / "embeddings.npz",
    "frame": SCRIPT_DIR / "data" / "embeddings_frame.npz",
}
DEFAULT_OUT = {
    "utterance": SCRIPT_DIR / "outputs" / "tsne_speaker_embeddings.png",
    "frame": SCRIPT_DIR / "outputs" / "tsne_speaker_embeddings_frame.png",
}
TITLES = {
    "utterance": "t-SNE of SileroVAD encoder embeddings (VIVOS speakers, mean-pooled per utterance)",
    "frame": "t-SNE of SileroVAD encoder embeddings (VIVOS speakers, one point per encoder frame)",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=["utterance", "frame"], default="utterance")
    parser.add_argument("--embeddings", type=Path, default=None,
                         help="Input .npz path. Defaults depend on --level.")
    parser.add_argument("--out", type=Path, default=None,
                         help="Output .png path. Defaults depend on --level.")
    return parser.parse_args()


def main():
    args = parse_args()
    emb_path = (args.embeddings or DEFAULT_EMBEDDINGS[args.level]).resolve()
    out_path = (args.out or DEFAULT_OUT[args.level]).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(emb_path, allow_pickle=True)
    embeddings = data["embeddings"]
    speakers = data["speakers"]

    n_samples = embeddings.shape[0]
    perplexity = min(30, max(5, n_samples // 4))
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=42)
    proj = tsne.fit_transform(embeddings)

    unique_speakers = sorted(set(speakers.tolist()))
    # tab20 alone only has 20 distinct hues; stitch tab20/tab20b/tab20c
    # together (60 distinct colors) so larger speaker counts stay legible,
    # cycling only if there are somehow more speakers than that.
    palette = [
        plt.get_cmap(name)(i)
        for name in ("tab20", "tab20b", "tab20c")
        for i in range(20)
    ]
    color_map = {spk: palette[i % len(palette)] for i, spk in enumerate(unique_speakers)}

    # Frame-level runs have far more points per speaker, so shrink markers
    # and add transparency to keep overlapping clusters legible.
    point_size = 40 if args.level == "utterance" else 8
    alpha = 0.9 if args.level == "utterance" else 0.4

    # A legend with many speakers needs more room and multiple columns.
    n_speakers = len(unique_speakers)
    legend_cols = 1 if n_speakers <= 25 else 2 if n_speakers <= 50 else 3
    fig_width = 8 + 1.6 * legend_cols
    fig_height = max(6, 0.16 * n_speakers)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    for spk in unique_speakers:
        mask = speakers == spk
        ax.scatter(proj[mask, 0], proj[mask, 1], color=color_map[spk], label=spk, s=point_size, alpha=alpha)

    ax.set_title(TITLES[args.level])
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.legend(title="speaker", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="x-small", ncol=legend_cols)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)

    # Save the 2D projection alongside the plot so downstream analysis (e.g.
    # clustering) can reuse the exact coordinates instead of recomputing t-SNE.
    proj_path = out_path.with_suffix(".proj.npz")
    np.savez(proj_path, proj=proj, speakers=speakers, utt_ids=data["utt_ids"])

    print(f"Saved plot ({n_samples} points, {len(unique_speakers)} speakers) -> {out_path}")
    print(f"Saved 2D projection -> {proj_path}")


if __name__ == "__main__":
    main()
