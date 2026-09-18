"""
Unit tests for BranchedEncoderBlock (silero/model_beautified.py):

1. from_reparam() produces a block whose forward() output is numerically
   identical to the original fused SileroVadEncoderBlock (zero-init branches).
2. After perturbing the new branches (simulating fine-tuning), reparameterize()
   folds them back into a single Conv1d whose forward() output matches the
   branched block's forward() output.
3. Sanity: reparameterize() fails to match if a branch is silently dropped
   (guards against a no-op fusion bug).
"""
import torch
import torch.nn as nn
import pytest

from silero.model_beautified import SileroVadEncoderBlock, BranchedEncoderBlock

torch.manual_seed(0)

BLOCK_SPECS = [
    # (in_channels, out_channels, stride) -- mirrors VADRNN.encoder in model_beautified.py
    (129, 128, 1),
    (128, 64, 2),
    (64, 64, 2),
    (64, 128, 1),
]


def _random_reparam_block(in_channels, out_channels, stride):
    block = SileroVadEncoderBlock(in_channels, out_channels, stride=stride)
    nn.init.normal_(block.reparam_conv.weight, std=0.5)
    nn.init.normal_(block.reparam_conv.bias, std=0.5)
    block.eval()
    return block


def _random_input(in_channels, length=17, batch=2):
    return torch.randn(batch, in_channels, length)


@pytest.mark.parametrize("in_channels,out_channels,stride", BLOCK_SPECS)
@pytest.mark.parametrize("use_identity", [True, False])
def test_from_reparam_matches_original_at_init(in_channels, out_channels, stride, use_identity):
    original = _random_reparam_block(in_channels, out_channels, stride)
    branched = BranchedEncoderBlock.from_reparam(original, use_identity=use_identity)
    branched.eval()

    x = _random_input(in_channels)
    with torch.no_grad():
        out_original = original(x)
        out_branched = branched(x)

    assert torch.allclose(out_original, out_branched, atol=1e-6), (
        f"BranchedEncoderBlock.from_reparam output diverges at init "
        f"(in={in_channels}, out={out_channels}, stride={stride}, use_identity={use_identity}); "
        f"max abs diff = {(out_original - out_branched).abs().max().item()}"
    )


@pytest.mark.parametrize("in_channels,out_channels,stride", BLOCK_SPECS)
@pytest.mark.parametrize("use_identity", [True, False])
def test_reparameterize_matches_branched_after_training_step(in_channels, out_channels, stride, use_identity):
    original = _random_reparam_block(in_channels, out_channels, stride)
    branched = BranchedEncoderBlock.from_reparam(original, use_identity=use_identity)

    # Simulate a step of fine-tuning: move every branch (including the BN
    # identity branch's affine params and running stats) away from its
    # zero-contribution initialization.
    branched.train()
    x = _random_input(in_channels, length=33, batch=4)
    out = branched(x)
    out.sum().backward()

    with torch.no_grad():
        for p in branched.parameters():
            if p.grad is not None:
                p -= 0.1 * p.grad
                p.grad = None

        if branched.has_identity:
            # push running stats away from the freshly-initialized (0, 1) so the
            # test actually exercises the running_mean/running_var fusion math
            branched.identity_bn.running_mean += 0.3
            branched.identity_bn.running_var *= 1.7

    branched.eval()
    fused_conv = branched.reparameterize()

    x_eval = _random_input(in_channels, length=25, batch=3)
    with torch.no_grad():
        # compare at the pre-(se+activation) point, which is where
        # reparameterize() claims equivalence to the branched sum.
        pre_act_branched = branched.conv3x1(x_eval) + branched.conv1x1(x_eval)
        if branched.has_identity:
            pre_act_branched = pre_act_branched + branched.identity_bn(x_eval)
        pre_act_fused = fused_conv(x_eval)

    # fused-then-convolve vs convolve-then-sum accumulate float32 rounding in a
    # different order, so allow a small relative tolerance on top of atol.
    assert torch.allclose(pre_act_branched, pre_act_fused, atol=1e-4, rtol=1e-4), (
        f"reparameterize() diverges from branched forward after a training step "
        f"(in={in_channels}, out={out_channels}, stride={stride}, use_identity={use_identity}); "
        f"max abs diff = {(pre_act_branched - pre_act_fused).abs().max().item()}"
    )

    # full block equivalence too (post se+activation)
    full_reparam_block = branched.to_reparam_block()
    full_reparam_block.eval()
    with torch.no_grad():
        out_branched_full = branched(x_eval)
        out_fused_full = full_reparam_block(x_eval)
    assert torch.allclose(out_branched_full, out_fused_full, atol=1e-4, rtol=1e-4)


def test_identity_branch_only_created_when_structurally_valid():
    # in != out -> no identity branch possible regardless of use_identity flag
    block = BranchedEncoderBlock(in_channels=64, out_channels=128, stride=1, use_identity=True)
    assert block.has_identity is False
    assert block.identity_bn is None

    # stride != 1 -> no identity branch possible even if in == out
    block = BranchedEncoderBlock(in_channels=64, out_channels=64, stride=2, use_identity=True)
    assert block.has_identity is False
    assert block.identity_bn is None

    # in == out, stride == 1 -> identity branch created (unless disabled)
    block = BranchedEncoderBlock(in_channels=64, out_channels=64, stride=1, use_identity=True)
    assert block.has_identity is True
    assert isinstance(block.identity_bn, nn.BatchNorm1d)

    block = BranchedEncoderBlock(in_channels=64, out_channels=64, stride=1, use_identity=False)
    assert block.has_identity is False


def test_reparam_conv_shape_and_type():
    original = _random_reparam_block(64, 64, 1)
    branched = BranchedEncoderBlock.from_reparam(original, use_identity=True)
    fused = branched.reparameterize()

    assert isinstance(fused, nn.Conv1d)
    assert fused.weight.shape == original.reparam_conv.weight.shape
    assert fused.bias.shape == original.reparam_conv.bias.shape
    assert fused.stride == original.reparam_conv.stride
    assert fused.padding == original.reparam_conv.padding
