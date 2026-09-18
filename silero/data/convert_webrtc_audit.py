"""
Convert a webrtc-audio-audit labeling_dataset (see
buddyos-omni-grpc/tools/webrtc-audio-audit/docs/LABELING.md for the source
schema) into this project's manifest format (data_format.md): one JSON object
per line, `{"filepath": ..., "segments": [{"start_ms": ..., "end_ms": ...}]}`.

Only `reviewed: true` entries are converted. Everything else in that dataset
is just the seed label from running the *current production* Silero VAD over
the recording -- not human-verified ground truth -- so training on it would
mean fine-tuning the model to reproduce (and reinforce) its own predictions,
including its own mistakes.

Only segments with `speech_type == "directed_speech"` are kept as positive
speech; segments typed `"echo"` / `"background_speech"`, or untyped (`null`),
are dropped (treated as background) -- per-project call: we only want the VAD
to fire on speech directed at the device, not incidental echo/background
chatter. Audio is transcoded from the source .opus (48kHz mono) to 16kHz mono
PCM wav, since that's what Silero VAD expects.

Usage:
    python -m silero.data.convert_webrtc_audit \
        --dataset-dir /path/to/webrtc-audio-audit/labeling_dataset \
        --out-dir egs/vad/webrtc_audit/data/webrtc_audit
"""
import argparse
import json
import subprocess
from pathlib import Path

import soundfile as sf

SAMPLE_RATE = 16000


def load_reviewed_entries(dataset_dir: Path) -> list[dict]:
    manifest_path = dataset_dir / "manifest.json"
    with open(manifest_path) as f:
        entries = json.load(f)

    reviewed = [e for e in entries if e.get("reviewed") and e.get("file_type") == "file"]
    return reviewed


def convert_audio(src: Path, dst: Path, sample_rate: int) -> bool:
    """Transcode src -> dst. Returns False (no-op) if dst is already cached,
    so re-running against a growing, ongoing dataset only pays for newly
    reviewed recordings."""
    if dst.exists() and dst.stat().st_size > 0:
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(src),
            "-ac", "1",
            "-ar", str(sample_rate),
            str(dst),
        ],
        check=True,
    )
    return True


def build_manifest_row(entry: dict, wav_path: Path, out_dir: Path) -> dict:
    segments = [
        {
            "start_ms": round(seg["start_microseconds"] / 1000),
            "end_ms": round(seg["end_microseconds"] / 1000),
        }
        for seg in entry["segments"]
        if seg.get("speech_type") == "directed_speech"
    ]
    info = sf.info(wav_path)
    duration_ms = round(info.frames / info.samplerate * 1000)
    return {
        # relative to the manifest's own directory, matching the convention
        # already used by silero/speaker_sim's manifest.csv.
        "filepath": str(wav_path.relative_to(out_dir)),
        "duration_ms": duration_ms,
        "segments": segments,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True,
                         help="webrtc-audio-audit labeling_dataset dir (contains manifest.json + audio/)")
    parser.add_argument("--out-dir", type=Path, required=True,
                         help="Destination dir; wav/ and manifest.jsonl are written here")
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    out_dir = args.out_dir.resolve()
    wav_dir = out_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    entries = load_reviewed_entries(dataset_dir)

    rows = []
    n_speech_segments = 0
    total_speech_ms = 0
    n_newly_converted = 0
    for entry in entries:
        src = dataset_dir / entry["file_path"]
        wav_path = wav_dir / f"{entry['id']}.wav"
        if convert_audio(src, wav_path, args.sample_rate):
            n_newly_converted += 1

        row = build_manifest_row(entry, wav_path, out_dir)
        rows.append(row)
        n_speech_segments += len(row["segments"])
        total_speech_ms += sum(s["end_ms"] - s["start_ms"] for s in row["segments"])

    manifest_path = out_dir / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    n_with_speech = sum(1 for r in rows if r["segments"])
    print(
        f"{len(rows)} reviewed recordings "
        f"({n_with_speech} with speech, {len(rows) - n_with_speech} confirmed silence), "
        f"{n_newly_converted} newly transcoded ({len(rows) - n_newly_converted} already cached) "
        f"-> {manifest_path}"
    )
    print(f"{n_speech_segments} speech segments, {total_speech_ms / 1000:.1f}s total speech")


if __name__ == "__main__":
    main()
