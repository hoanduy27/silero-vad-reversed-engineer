"""
Unit tests for silero/trainer/trainer.py's model-building and freeze logic.
Uses randomly-initialized SileroVadEncoderBlock/Decoder stand-ins (not the
real checkpoint) so these run without assets/silero_vad.jit.
"""
import pytest
import torch
import torch.nn as nn

from silero.data.dataset import collate_batch
from silero.model_beautified import BranchedEncoderBlock, Decoder, VADRNN
from silero.trainer.trainer import (
    METRIC_MODES,
    EarlyStopper,
    apply_freeze,
    average_state_dicts,
    branch_encoder,
    compute_loss,
    reparameterize_model,
    run_epoch,
)


def _tiny_vadrnn() -> VADRNN:
    # VADRNN() already builds the real 4-block encoder + Decoder with random
    # init (no checkpoint loading needed to exercise branching/freezing).
    return VADRNN()


def test_branch_encoder_replaces_all_blocks_with_branched():
    model = _tiny_vadrnn()
    model = branch_encoder(model, use_identity=True)
    assert all(isinstance(block, BranchedEncoderBlock) for block in model.encoder)


@pytest.mark.parametrize("freeze,expect_conv3x1_trainable,expect_decoder_trainable", [
    ([], True, True),
    (["conv3x1"], False, True),
    (["decoder"], True, False),
    (["conv3x1", "decoder"], False, False),
])
def test_apply_freeze_sets_requires_grad_correctly(freeze, expect_conv3x1_trainable, expect_decoder_trainable):
    model = branch_encoder(_tiny_vadrnn())
    apply_freeze(model, freeze)

    for block in model.encoder:
        assert all(p.requires_grad == expect_conv3x1_trainable for p in block.conv3x1.parameters())
        # conv1x1 (the actual adapter branch) must never be frozen by these configs
        assert all(p.requires_grad for p in block.conv1x1.parameters())

    assert all(p.requires_grad == expect_decoder_trainable for p in model.decoder.parameters())


def test_apply_freeze_unknown_target_raises():
    model = branch_encoder(_tiny_vadrnn())
    with pytest.raises(ValueError):
        apply_freeze(model, ["not_a_real_target"])


def test_reparameterize_model_matches_branched_model_output():
    model = branch_encoder(_tiny_vadrnn())
    # perturb an adapter branch, like a step of training would
    with torch.no_grad():
        model.encoder[0].conv1x1.weight.add_(0.05)
        model.encoder[0].conv1x1.bias.add_(0.05)
    model.eval()

    fused = reparameterize_model(model)
    assert all(not isinstance(b, BranchedEncoderBlock) for b in fused.encoder)

    x = torch.randn(1, 512 * 3)
    with torch.no_grad():
        out_branched = model.forward_sequence(x)
        out_fused = fused.forward_sequence(x)
    assert torch.allclose(out_branched, out_fused, atol=1e-5, rtol=1e-4)


def test_reparameterize_model_does_not_mutate_input_model():
    model = branch_encoder(_tiny_vadrnn())
    original_weight = model.encoder[0].conv3x1.weight.clone()
    reparameterize_model(model)
    assert torch.equal(model.encoder[0].conv3x1.weight, original_weight)
    assert isinstance(model.encoder[0], BranchedEncoderBlock)


def test_early_stopper_does_not_stop_while_improving():
    stopper = EarlyStopper(patience=3)
    for val_loss in [1.0, 0.9, 0.8, 0.7]:
        assert stopper.step(val_loss) is False


def test_early_stopper_stops_after_patience_epochs_without_improvement():
    stopper = EarlyStopper(patience=3)
    assert stopper.step(1.0) is False   # epoch 0: new best
    assert stopper.step(1.1) is False   # 1 epoch without improvement
    assert stopper.step(1.2) is False   # 2
    assert stopper.step(1.3) is True    # 3 -> stop


def test_early_stopper_min_delta_requires_meaningful_improvement():
    stopper = EarlyStopper(patience=2, min_delta=0.1)
    assert stopper.step(1.0) is False    # new best
    assert stopper.step(0.95) is False   # improved, but by < min_delta -> doesn't count
    assert stopper.step(0.95) is True    # 2nd epoch without a >=min_delta improvement -> stop


def test_early_stopper_resets_patience_on_real_improvement():
    stopper = EarlyStopper(patience=2)
    assert stopper.step(1.0) is False
    assert stopper.step(1.1) is False   # 1 without improvement
    assert stopper.step(0.5) is False   # improves -> resets counter
    assert stopper.step(0.6) is False   # 1 without improvement again
    assert stopper.step(0.6) is True    # 2 -> stop


def test_average_state_dicts_computes_elementwise_mean():
    sds = [
        {"w": torch.tensor([1.0, 2.0]), "b": torch.tensor(0.0)},
        {"w": torch.tensor([3.0, 4.0]), "b": torch.tensor(2.0)},
    ]
    averaged = average_state_dicts(sds)
    assert torch.allclose(averaged["w"], torch.tensor([2.0, 3.0]))
    assert torch.allclose(averaged["b"], torch.tensor(1.0))


def test_average_state_dicts_single_checkpoint_is_identity():
    sd = {"w": torch.tensor([1.0, 2.0, 3.0])}
    averaged = average_state_dicts([sd])
    assert torch.allclose(averaged["w"], sd["w"])


def test_average_state_dicts_preserves_dtype():
    sds = [
        {"w": torch.tensor([1.0, 2.0], dtype=torch.float32)},
        {"w": torch.tensor([3.0, 4.0], dtype=torch.float32)},
    ]
    averaged = average_state_dicts(sds)
    assert averaged["w"].dtype == torch.float32


def _fake_dataset_item(num_frames, seed):
    g = torch.Generator().manual_seed(seed)
    return {
        "audio": torch.rand(num_frames * 512, generator=g),
        "target": (torch.rand(num_frames, generator=g) > 0.5).float(),
    }


def test_batched_forward_matches_single_item_forward_on_the_valid_region():
    """The whole point of padding+masking: a short item's own valid-frame
    predictions must be identical whether it runs alone or padded inside a
    batch with a longer item -- padding must never leak across the batch dim."""
    torch.manual_seed(0)
    model = _tiny_vadrnn()
    model.eval()

    short_item = _fake_dataset_item(num_frames=3, seed=1)
    long_item = _fake_dataset_item(num_frames=7, seed=2)

    with torch.no_grad():
        alone = model.forward_sequence(short_item["audio"].unsqueeze(0)).squeeze(0)

        batch = collate_batch([short_item, long_item])
        batched = model.forward_sequence(batch["audio"])  # (2, 7)
        short_in_batch = batched[0, :3]

    assert torch.allclose(alone, short_in_batch, atol=1e-6)


def test_compute_loss_ignores_padded_frames():
    torch.manual_seed(0)
    model = _tiny_vadrnn()

    short_item = _fake_dataset_item(num_frames=2, seed=1)
    long_item = _fake_dataset_item(num_frames=5, seed=2)
    batch = collate_batch([short_item, long_item])

    # sanity: valid_frames correctly recorded
    assert batch["valid_frames"].tolist() == [2, 5]

    result = compute_loss(model, batch, torch.device("cpu"))
    assert result is not None
    assert result["n"] == 2 + 5  # not 2*5 -- padded (2..5) frames of the short item excluded
    assert result["loss"] is not None
    assert sum(result["counts"].values()) == 2 + 5


def test_metric_modes_covers_every_run_epoch_key():
    assert set(METRIC_MODES) == {"loss", "acc", "precision", "recall", "f1"}
    assert METRIC_MODES["loss"] == "min"
    assert METRIC_MODES["f1"] == "max"


def test_early_stopper_max_mode_stops_when_metric_stops_increasing():
    stopper = EarlyStopper(patience=2, mode="max")
    assert stopper.step(0.5) is False   # new best
    assert stopper.step(0.6) is False   # improved -> resets
    assert stopper.step(0.55) is False  # 1 without improvement
    assert stopper.step(0.55) is True   # 2 -> stop


def test_early_stopper_max_mode_min_delta():
    stopper = EarlyStopper(patience=2, min_delta=0.05, mode="max")
    assert stopper.step(0.5) is False    # new best
    assert stopper.step(0.52) is False   # improved, but < min_delta -> doesn't count
    assert stopper.step(0.52) is True    # 2nd epoch without a >=min_delta improvement -> stop


def _write_wav_row(tmp_path, name, num_frames, speech_frame_indices=()):
    import numpy as np
    import soundfile as sf
    from silero.data.dataset import SAMPLES_PER_FRAME

    audio = np.random.randn(num_frames * SAMPLES_PER_FRAME).astype(np.float32)
    sf.write(tmp_path / name, audio, 16000)

    frame_ms = SAMPLES_PER_FRAME / 16000 * 1000
    segments = [
        {"start_ms": round(i * frame_ms), "end_ms": round((i + 1) * frame_ms)}
        for i in speech_frame_indices
    ]
    return {"filepath": name, "duration_ms": round(num_frames * frame_ms), "segments": segments}


def test_run_epoch_returns_full_metrics_dict(tmp_path):
    rows = [
        _write_wav_row(tmp_path, "a.wav", 4, speech_frame_indices=[1, 2]),
        _write_wav_row(tmp_path, "b.wav", 6, speech_frame_indices=[]),
    ]
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text("")

    model = branch_encoder(_tiny_vadrnn())
    metrics = run_epoch(model, rows, manifest_path, torch.device("cpu"), batch_size=2)

    assert set(metrics) == {"loss", "acc", "precision", "recall", "f1"}
    assert 0.0 <= metrics["acc"] <= 1.0
    assert metrics["loss"] == metrics["loss"]  # not nan, given non-empty rows
