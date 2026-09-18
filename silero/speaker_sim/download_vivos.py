"""
Download the VIVOS Vietnamese speech corpus and build a small subset
(a handful of utterances for a handful of speakers) to use for speaker
embedding experiments.

Usage:
    python -m silero.speaker_sim.download_vivos --out-dir exp/data
"""
import argparse
import csv
import os
import tarfile
from pathlib import Path

import requests

VIVOS_URL = "https://huggingface.co/datasets/AILAB-VNUHCM/vivos/resolve/main/data/vivos.tar.gz"

SCRIPT_DIR = Path(__file__).resolve().parent

# Where the full archive/extraction is cached (kept out of any experiment dir,
# it's ~1.4GB/2GB and reusable across runs).
CACHE_DIR = Path(os.environ.get("VIVOS_CACHE_DIR", "/tmp/vivos_cache"))

SPLIT = "train"
NUM_SPEAKERS = 10
UTTS_PER_SPEAKER = 5
DEFAULT_OUT_DIR = SCRIPT_DIR / "data"


def download_archive(dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        print(f"Using cached archive at {dst}")
        return dst

    print(f"Downloading {VIVOS_URL} -> {dst}")
    with requests.get(VIVOS_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        written = 0
        tmp_dst = dst.with_suffix(dst.suffix + ".part")
        with open(tmp_dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                written += len(chunk)
                if total:
                    print(f"\r  {written / 1e6:8.1f} / {total / 1e6:8.1f} MB", end="")
        print()
        tmp_dst.rename(dst)
    return dst


def extract_archive(archive: Path, out_dir: Path) -> Path:
    root = out_dir / "vivos"
    if root.exists():
        print(f"Already extracted at {root}")
        return root

    print(f"Extracting {archive} -> {out_dir}")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(out_dir)
    return root


def build_subset(vivos_root: Path, split: str, num_speakers: int, utts_per_speaker: int, out_dir: Path) -> Path:
    waves_dir = vivos_root / split / "waves"
    speakers = sorted(p.name for p in waves_dir.iterdir() if p.is_dir())[:num_speakers]

    subset_dir = out_dir / "vivos_subset"
    subset_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"

    rows = []
    for speaker in speakers:
        wavs = sorted((waves_dir / speaker).glob("*.wav"))[:utts_per_speaker]
        out_speaker_dir = subset_dir / speaker
        out_speaker_dir.mkdir(parents=True, exist_ok=True)
        for wav in wavs:
            dst = out_speaker_dir / wav.name
            if not dst.exists():
                dst.write_bytes(wav.read_bytes())
            rows.append({
                "utt_id": wav.stem,
                "speaker": speaker,
                # relative to the manifest's own directory, so (out_dir, manifest.csv,
                # vivos_subset/) stay portable together regardless of where out_dir lives.
                "path": str(dst.relative_to(out_dir)),
            })

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["utt_id", "speaker", "path"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} utterances from {len(speakers)} speakers -> {manifest_path}")
    return manifest_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default=SPLIT, choices=["train", "test"])
    parser.add_argument("--num-speakers", type=int, default=NUM_SPEAKERS)
    parser.add_argument("--utts-per-speaker", type=int, default=UTTS_PER_SPEAKER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                         help="Directory to write manifest.csv and vivos_subset/ into.")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    archive = download_archive(CACHE_DIR / "vivos.tar.gz")
    vivos_root = extract_archive(archive, CACHE_DIR / "extracted")
    build_subset(vivos_root, args.split, args.num_speakers, args.utts_per_speaker, out_dir)


if __name__ == "__main__":
    main()
