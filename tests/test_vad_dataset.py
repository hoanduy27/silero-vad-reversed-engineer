"""Unit tests for silero/data/dataset.py's labeling, batching, and split logic."""
import numpy as np
import soundfile as sf
import torch

from silero.data.dataset import VadJsonlDataset, collate_batch, frame_targets, make_batches, split_train_val

SAMPLE_RATE = 16000
SAMPLES_PER_FRAME = 512
FRAME_MS = SAMPLES_PER_FRAME / SAMPLE_RATE * 1000  # 32ms/frame


def test_frame_targets_no_segments_is_all_zero():
    num_samples = SAMPLES_PER_FRAME * 10
    target = frame_targets([], num_samples)
    assert target.shape == (10,)
    assert torch.all(target == 0)


def test_frame_targets_any_overlap_marks_frame_positive():
    num_samples = SAMPLES_PER_FRAME * 4
    # segment covers only the tail of frame 1 (1ms before its end) -- "any
    # overlap" should still mark frame 1 positive, not just frame 2.
    frame1_end_ms = round(2 * FRAME_MS)
    segments = [{"start_ms": frame1_end_ms - 1, "end_ms": frame1_end_ms + 5}]
    target = frame_targets(segments, num_samples)
    assert target.tolist() == [0.0, 1.0, 1.0, 0.0]


def test_frame_targets_covers_full_segment_span():
    num_samples = SAMPLES_PER_FRAME * 5
    segments = [{"start_ms": 0, "end_ms": round(3 * FRAME_MS)}]
    target = frame_targets(segments, num_samples)
    assert target.tolist() == [1.0, 1.0, 1.0, 0.0, 0.0]


def test_frame_targets_multiple_segments_union():
    num_samples = SAMPLES_PER_FRAME * 6
    segments = [
        {"start_ms": 0, "end_ms": round(FRAME_MS)},
        {"start_ms": round(4 * FRAME_MS), "end_ms": round(5 * FRAME_MS)},
    ]
    target = frame_targets(segments, num_samples)
    assert target.tolist() == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def test_frame_targets_segment_beyond_truncated_length_is_clamped():
    num_samples = SAMPLES_PER_FRAME * 2
    # end_ms far beyond the (already truncated-to-whole-frames) audio length
    segments = [{"start_ms": 0, "end_ms": round(100 * FRAME_MS)}]
    target = frame_targets(segments, num_samples)
    assert target.shape == (2,)
    assert torch.all(target == 1.0)


def _row(has_speech: bool, idx: int):
    return {
        "filepath": f"wav/{idx}.wav",
        "duration_ms": 1000,
        "segments": [{"start_ms": 0, "end_ms": 100}] if has_speech else [],
    }


def test_split_train_val_is_stratified_by_speech_presence():
    rows = [_row(True, i) for i in range(20)] + [_row(False, i) for i in range(10)]
    train, val = split_train_val(rows, val_fraction=0.2, seed=0)

    assert len(train) + len(val) == len(rows)
    val_speech = sum(1 for r in val if r["segments"])
    val_silence = sum(1 for r in val if not r["segments"])
    # 20*0.2=4 speech, 10*0.2=2 silence
    assert val_speech == 4
    assert val_silence == 2


def test_split_train_val_is_deterministic_given_seed():
    rows = [_row(i % 3 == 0, i) for i in range(30)]
    train1, val1 = split_train_val(rows, val_fraction=0.2, seed=42)
    train2, val2 = split_train_val(rows, val_fraction=0.2, seed=42)
    assert [r["filepath"] for r in val1] == [r["filepath"] for r in val2]
    assert [r["filepath"] for r in train1] == [r["filepath"] for r in train2]


def test_split_train_val_no_overlap():
    rows = [_row(i % 2 == 0, i) for i in range(30)]
    train, val = split_train_val(rows, val_fraction=0.2, seed=0)
    train_paths = {r["filepath"] for r in train}
    val_paths = {r["filepath"] for r in val}
    assert train_paths.isdisjoint(val_paths)


def _item(num_frames, value):
    return {
        "audio": torch.full((num_frames * SAMPLES_PER_FRAME,), value),
        "target": torch.full((num_frames,), float(value % 2)),
    }


def test_collate_batch_pads_to_longest_item():
    items = [_item(3, 1), _item(5, 2)]
    batch = collate_batch(items)
    assert batch["audio"].shape == (2, 5 * SAMPLES_PER_FRAME)
    assert batch["target"].shape == (2, 5)
    assert batch["valid_frames"].tolist() == [3, 5]


def test_collate_batch_padding_is_zero_and_valid_region_is_untouched():
    items = [_item(2, 7), _item(4, 9)]
    batch = collate_batch(items)

    # item 0: valid audio/target in [:2*SAMPLES_PER_FRAME]/[:2], zero padding after
    assert torch.all(batch["audio"][0, :2 * SAMPLES_PER_FRAME] == 7)
    assert torch.all(batch["audio"][0, 2 * SAMPLES_PER_FRAME:] == 0)
    assert torch.all(batch["target"][0, :2] == 1.0)  # 7 % 2 == 1
    assert torch.all(batch["target"][0, 2:] == 0)

    # item 1 fills the full (longest) length -- no padding
    assert torch.all(batch["audio"][1] == 9)
    assert torch.all(batch["target"][1] == 1.0)  # 9 % 2 == 1


def test_collate_batch_single_item_is_unpadded():
    items = [_item(4, 3)]
    batch = collate_batch(items)
    assert batch["audio"].shape == (1, 4 * SAMPLES_PER_FRAME)
    assert batch["valid_frames"].tolist() == [4]


def _duration_row(idx, duration_ms):
    return {"filepath": f"wav/{idx}.wav", "duration_ms": duration_ms, "segments": []}


def test_make_batches_groups_by_duration():
    # durations 400, 100, 300, 200 at indices 0..3 -> sorted order should be 1,3,2,0
    rows = [_duration_row(0, 400), _duration_row(1, 100), _duration_row(2, 300), _duration_row(3, 200)]
    batches = make_batches(rows, batch_size=2, shuffle=False)
    assert batches == [[1, 3], [2, 0]]


def test_make_batches_covers_every_row_exactly_once():
    rows = [_duration_row(i, i * 37 % 500) for i in range(23)]
    batches = make_batches(rows, batch_size=4, shuffle=False)
    all_indices = sorted(i for batch in batches for i in batch)
    assert all_indices == list(range(23))


def test_make_batches_shuffle_only_reorders_batches_not_contents():
    rows = [_duration_row(i, i * 37 % 500) for i in range(23)]
    unshuffled = make_batches(rows, batch_size=4, shuffle=False)
    shuffled = make_batches(rows, batch_size=4, shuffle=True, seed=1)

    assert sorted(map(tuple, shuffled)) == sorted(map(tuple, unshuffled))
    assert shuffled != unshuffled  # extremely unlikely to coincide by chance


def test_make_batches_shuffle_is_deterministic_given_seed():
    rows = [_duration_row(i, i * 37 % 500) for i in range(23)]
    a = make_batches(rows, batch_size=4, shuffle=True, seed=7)
    b = make_batches(rows, batch_size=4, shuffle=True, seed=7)
    assert a == b


def test_max_frames_truncates_long_utterances(tmp_path):
    n_frames_full = 20
    audio = np.random.randn(n_frames_full * SAMPLES_PER_FRAME).astype(np.float32)
    wav_path = tmp_path / "long.wav"
    sf.write(wav_path, audio, SAMPLE_RATE)

    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text("")  # only manifest_path.parent matters; rows passed explicitly
    row = {"filepath": "long.wav", "duration_ms": 0, "segments": []}

    full = VadJsonlDataset(manifest_path, rows=[row])[0]
    assert full["target"].numel() == n_frames_full

    capped = VadJsonlDataset(manifest_path, rows=[row], max_frames=5)[0]
    assert capped["target"].numel() == 5
    assert capped["audio"].numel() == 5 * SAMPLES_PER_FRAME
    assert torch.equal(capped["audio"], full["audio"][:5 * SAMPLES_PER_FRAME])


def test_max_frames_is_a_noop_when_utterance_is_already_shorter(tmp_path):
    audio = np.random.randn(3 * SAMPLES_PER_FRAME).astype(np.float32)
    wav_path = tmp_path / "short.wav"
    sf.write(wav_path, audio, SAMPLE_RATE)

    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text("")
    row = {"filepath": "short.wav", "duration_ms": 0, "segments": []}

    capped = VadJsonlDataset(manifest_path, rows=[row], max_frames=100)[0]
    assert capped["target"].numel() == 3
