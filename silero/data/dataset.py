"""
Dataset for fine-tuning VADRNN on the manifest format in data_format.md: one
JSON object per line, `{"filepath": ..., "segments": [{"start_ms", "end_ms"}]}`.

Each item is a full utterance, processed chunk-by-chunk (VADRNN.forward_sequence
replicates the real streaming computation exactly, see its docstring) against a
per-chunk binary speech/silence target built from the manifest's segments.
`make_batches`/`collate_batch` pad same-batch utterances to equal length (in
whole chunks) and track each one's true length, so silero.trainer.trainer can
process several utterances per forward/backward pass instead of one at a time
-- forward_sequence already operates over an arbitrary batch dimension, so
padding/masking here is the only piece needed for real (non-accumulated)
batching.
"""
import json
import random
from pathlib import Path
from typing import Optional

import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset

# Encoder output frame -> input samples: STFT hop_length=128 x encoder strides
# [1, 2, 2, 1] (total 4x downsample) = 512 samples/frame, matching the
# `num_samples=512` chunk size used for 16kHz streaming inference.
SAMPLES_PER_FRAME = 512


def load_manifest(manifest_path: Path) -> list[dict]:
    rows = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def frame_targets(segments: list[dict], num_samples: int, sample_rate: int = 16000,
                   samples_per_frame: int = SAMPLES_PER_FRAME) -> Tensor:
    """
    Binary target per encoder output frame: 1.0 if the frame's sample window
    overlaps a labeled speech segment at all, else 0.0 ("any overlap" rule).
    """
    num_frames = num_samples // samples_per_frame
    target = torch.zeros(num_frames)

    for seg in segments:
        start_sample = int(seg["start_ms"] * sample_rate / 1000)
        end_sample = int(seg["end_ms"] * sample_rate / 1000)

        first_frame = max(0, start_sample // samples_per_frame)
        last_frame = min(num_frames - 1, (end_sample - 1) // samples_per_frame if end_sample > 0 else -1)
        if first_frame <= last_frame:
            target[first_frame:last_frame + 1] = 1.0

    return target


class VadJsonlDataset(Dataset):
    """
    manifest_path: path to a manifest.jsonl file. `filepath` in each row is
    resolved relative to the manifest's own directory.

    max_frames: if set, utterances longer than this are truncated to their
    first `max_frames` chunks. forward_sequence backpropagates through the
    entire sequence (full BPTT, no truncation internally), so a handful of
    very long recordings (this dataset ranges from under a second to over 13
    minutes) can exhaust GPU memory during training even at batch_size=1 --
    this bounds that. Only matters for training; eval/export run forward-only
    under torch.no_grad(), which doesn't retain per-step activations, so
    arbitrarily long utterances are cheap there regardless.
    """

    def __init__(self, manifest_path: Path, sample_rate: int = 16000,
                 samples_per_frame: int = SAMPLES_PER_FRAME, rows: Optional[list[dict]] = None,
                 max_frames: Optional[int] = None):
        self.manifest_path = Path(manifest_path)
        self.manifest_dir = self.manifest_path.parent
        self.rows = rows if rows is not None else load_manifest(self.manifest_path)
        self.sample_rate = sample_rate
        self.samples_per_frame = samples_per_frame
        self.max_frames = max_frames

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        wav_path = self.manifest_dir / row["filepath"]
        audio, sr = sf.read(wav_path, dtype="float32")
        assert sr == self.sample_rate, f"Expected {self.sample_rate}Hz audio, got {sr} for {wav_path}"

        # truncate to a whole number of frames so the target aligns exactly
        # with the encoder's output length (and further to max_frames, if set).
        num_frames = len(audio) // self.samples_per_frame
        if self.max_frames is not None:
            num_frames = min(num_frames, self.max_frames)
        num_samples = num_frames * self.samples_per_frame
        audio = audio[:num_samples]

        target = frame_targets(row["segments"], num_samples, self.sample_rate, self.samples_per_frame)

        return {
            "audio": torch.from_numpy(audio),  # (num_samples,)
            "target": target,                    # (num_frames,)
            "filepath": row["filepath"],
        }


def collate_batch(items: list[dict]) -> dict:
    """
    Pad a list of VadJsonlDataset items (variable-length audio/target) to the
    longest item's length (in whole SAMPLES_PER_FRAME chunks), and record each
    item's true frame count so the trainer can mask padded frames out of the
    loss -- padding never leaks into another item's result since the batch
    dimension is independent all the way through the model (Conv1d/LSTMCell),
    only the shared Python chunk-loop length is affected.
    """
    max_frames = max(item["target"].numel() for item in items)
    max_samples = max_frames * SAMPLES_PER_FRAME

    batch_audio = torch.zeros(len(items), max_samples)
    batch_target = torch.zeros(len(items), max_frames)
    valid_frames = torch.zeros(len(items), dtype=torch.long)

    for i, item in enumerate(items):
        n_samples = item["audio"].numel()
        n_frames = item["target"].numel()
        batch_audio[i, :n_samples] = item["audio"]
        batch_target[i, :n_frames] = item["target"]
        valid_frames[i] = n_frames

    return {"audio": batch_audio, "target": batch_target, "valid_frames": valid_frames}


def make_batches(rows: list[dict], batch_size: int, shuffle: bool = False, seed: int = 0) -> list:
    """
    Group `rows` indices into batches of up to `batch_size`, bucketed by
    duration first (so same-batch utterances are similar length -- minimizes
    wasted padding compute and keeps the shared chunk-loop length close to
    each item's own length) and, if `shuffle`, with batch *order* then
    shuffled (composition of each batch stays duration-sorted; only which
    batch comes first varies). Returns index batches (into `rows`), not the
    rows themselves, so callers can pair them directly with a Dataset built
    from the same `rows` list without any row-identity bookkeeping.
    """
    order = sorted(range(len(rows)), key=lambda i: rows[i].get("duration_ms", 0))
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]

    if shuffle:
        random.Random(seed).shuffle(batches)

    return batches


def split_train_val(rows: list[dict], val_fraction: float, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """
    Deterministic file-level split, stratified by whether the file has any
    speech segments, so val isn't accidentally all-silence or all-speech.
    """
    with_speech = [r for r in rows if r["segments"]]
    without_speech = [r for r in rows if not r["segments"]]

    rng = random.Random(seed)
    rng.shuffle(with_speech)
    rng.shuffle(without_speech)

    def split(group):
        n_val = round(len(group) * val_fraction)
        return group[n_val:], group[:n_val]

    train_speech, val_speech = split(with_speech)
    train_silence, val_silence = split(without_speech)

    return train_speech + train_silence, val_speech + val_silence
