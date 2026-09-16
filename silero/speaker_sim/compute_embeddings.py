"""
Compute speaker embeddings using only the SileroVAD encoder
(VADRNN.forward_embed), skipping the decoder RNN entirely.

Two levels are supported:
- utterance: one mean-pooled embedding per utterance (forward_embed default).
- frame: every encoder time step kept as its own embedding
  (forward_embed(..., return_individuals=True)), tagged with its parent
  utterance's speaker.

Usage:
    python -m silero.speaker_sim.compute_embeddings --level utterance \
        --manifest exp/data/manifest.csv --out exp/embeddings/embeddings.npz
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from silero.model_beautified import VADRNN  # noqa: E402

DEFAULT_MANIFEST = SCRIPT_DIR / "data" / "manifest.csv"
JIT_PATH = REPO_ROOT / "assets" / "silero_vad.jit"

DEFAULT_OUT = {
    "utterance": SCRIPT_DIR / "data" / "embeddings.npz",
    "frame": SCRIPT_DIR / "data" / "embeddings_frame.npz",
}


def load_model() -> VADRNN:
    model = VADRNN()
    loaded = torch.jit.load(str(JIT_PATH))
    state_dict = loaded._model.state_dict()
    state_dict.pop("stft.forward_basis_buffer", None)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def compute_utterance_embeddings(model, rows, manifest_dir: Path):
    embeddings, speakers, utt_ids = [], [], []

    with torch.no_grad():
        for row in rows:
            wav_path = manifest_dir / row["path"]
            audio, sr = sf.read(wav_path, dtype="float32")
            assert sr == 16000, f"Expected 16kHz audio, got {sr} for {wav_path}"

            x = torch.from_numpy(audio).unsqueeze(0)  # (1, samples)
            emb = model.forward_embed(x)  # (1, 128)

            embeddings.append(emb.squeeze(0).numpy())
            speakers.append(row["speaker"])
            utt_ids.append(row["utt_id"])

    return np.stack(embeddings), np.array(speakers), np.array(utt_ids)


def compute_frame_embeddings(model, rows, manifest_dir: Path):
    embeddings, speakers, utt_ids = [], [], []

    with torch.no_grad():
        for row in rows:
            wav_path = manifest_dir / row["path"]
            audio, sr = sf.read(wav_path, dtype="float32")
            assert sr == 16000, f"Expected 16kHz audio, got {sr} for {wav_path}"

            x = torch.from_numpy(audio).unsqueeze(0)  # (1, samples)
            emb = model.forward_embed(x, return_individuals=True)  # (1, frames, 128)
            emb = emb.squeeze(0).numpy()  # (frames, 128)

            for frame_idx in range(emb.shape[0]):
                embeddings.append(emb[frame_idx])
                speakers.append(row["speaker"])
                utt_ids.append(f"{row['utt_id']}_f{frame_idx:03d}")

    return np.stack(embeddings), np.array(speakers), np.array(utt_ids)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=["utterance", "frame"], default="utterance")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=None,
                         help="Output .npz path. Defaults depend on --level.")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path = args.manifest.resolve()
    out_path = (args.out or DEFAULT_OUT[args.level]).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_model()

    with open(manifest_path) as f:
        rows = list(csv.DictReader(f))
    manifest_dir = manifest_path.parent

    if args.level == "utterance":
        embeddings, speakers, utt_ids = compute_utterance_embeddings(model, rows, manifest_dir)
    else:
        embeddings, speakers, utt_ids = compute_frame_embeddings(model, rows, manifest_dir)

    np.savez(out_path, embeddings=embeddings, speakers=speakers, utt_ids=utt_ids)
    print(
        f"Saved {embeddings.shape} '{args.level}'-level embeddings for "
        f"{len(set(speakers.tolist()))} speakers -> {out_path}"
    )


if __name__ == "__main__":
    main()
