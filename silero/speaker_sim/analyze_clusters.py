"""
Check whether visual clusters in a t-SNE plot (from plot_tsne.py) correspond
to distinct speaker groups, or whether every cluster is just a mix of the
same speakers.

Fits KMeans on the saved 2D t-SNE projection (not the raw embeddings -- we
want to know about the clusters actually visible in the plot), then reports,
per cluster, how many of its points belong to each speaker, and per speaker,
how its utterances split across clusters.

Usage:
    python -m silero.speaker_sim.analyze_clusters --level utterance --n-clusters 2
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_PROJ = {
    "utterance": SCRIPT_DIR / "outputs" / "tsne_speaker_embeddings.proj.npz",
    "frame": SCRIPT_DIR / "outputs" / "tsne_speaker_embeddings_frame.proj.npz",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=["utterance", "frame"], default="utterance")
    parser.add_argument("--proj", type=Path, default=None,
                         help="Path to a *.proj.npz saved by plot_tsne.py. Defaults depend on --level.")
    parser.add_argument("--n-clusters", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    proj_path = (args.proj or DEFAULT_PROJ[args.level]).resolve()

    data = np.load(proj_path, allow_pickle=True)
    proj = data["proj"]
    speakers = data["speakers"]

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=10, random_state=42)
    cluster_ids = kmeans.fit_predict(proj)

    unique_speakers = sorted(set(speakers.tolist()))
    unique_clusters = sorted(set(cluster_ids.tolist()))

    print(f"{proj.shape[0]} points, {len(unique_speakers)} speakers, {len(unique_clusters)} clusters\n")

    # Cluster -> speaker composition.
    print("Cluster composition (speakers present, by point count):")
    for c in unique_clusters:
        mask = cluster_ids == c
        speaker_counts = Counter(speakers[mask].tolist())
        n_points = mask.sum()
        n_spk = len(speaker_counts)
        top = ", ".join(f"{spk}={cnt}" for spk, cnt in speaker_counts.most_common(8))
        more = "" if n_spk <= 8 else f", ... (+{n_spk - 8} more speakers)"
        print(f"  cluster {c}: {n_points} points, {n_spk} distinct speakers -> {top}{more}")

    # Speaker -> cluster split: is each speaker confined to one cluster, or spread across both?
    print("\nPer-speaker cluster split (fraction of that speaker's points in each cluster):")
    n_confined = 0
    for spk in unique_speakers:
        mask = speakers == spk
        counts = Counter(cluster_ids[mask].tolist())
        total = mask.sum()
        fracs = {c: counts.get(c, 0) / total for c in unique_clusters}
        dominant_frac = max(fracs.values())
        if dominant_frac >= 0.9:
            n_confined += 1
        frac_str = ", ".join(f"c{c}={fracs[c]:.0%}" for c in unique_clusters)
        print(f"  {spk}: {frac_str}")

    n_speakers = len(unique_speakers)
    print(
        f"\n{n_confined}/{n_speakers} speakers have >=90% of their points in a single cluster."
    )
    if n_confined <= 0.2 * n_speakers:
        print("-> Clusters do NOT align with speaker identity; every cluster shares the same speakers.")
    elif n_confined >= 0.8 * n_speakers:
        print("-> Clusters largely DO align with speaker identity.")
    else:
        print("-> Mixed: some speakers are cluster-confined, others are split across clusters.")


if __name__ == "__main__":
    main()
