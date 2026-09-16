# Reverse-engineering Silero VAD

Goal: make `silero/model_beautified.py` (the human-written reconstruction) numerically
match `assets/silero_vad.jit` (the original TorchScript model), guided by the decompiled
forward methods in `silero/model_reversed.py`.

## Bug 1 — full-output mismatch despite matching sub-modules

`silero/validate.py` checks each sub-module of the model independently
(`compute_error_prenet` for the STFT, `compute_error_encoder`, and
`compute_error_decoder_*`), then compares the full `VAD.forward` output end to end.
With weights loaded via `load_state_dict`, every sub-module check reported `0` error,
but the full pipeline output was clearly different between the JIT model and the
reversed model (`0.0077` vs `0.0006`).

Since the sub-modules matched in isolation but not when composed, the bug had to be in
how they were wired together — shapes, strides, or context handling — not in the
weights themselves.

Feeding identical, context-prefixed input through both models step by step showed the
divergence appears right after the encoder: the JIT model's `stft` output shape
matched ours exactly, but its `encoder` output shape did not — `(1, 128, 9)` vs our
`(1, 128, 35)`. Our `SileroVadEncoderBlock` always used `stride=1`, so the frame axis
was never downsampled.

`Conv1d.stride` is a module attribute set once at construction time — it never shows
up as an op in TorchScript's decompiled `.code`, so it has to be recovered by
inspecting the live submodules directly:

```python
for name, m in loaded._model.encoder.named_children():
    conv = m.reparam_conv
    print(name, conv.stride)
# 0 (1,)   1 (2,)   2 (2,)   3 (1,)
```

The real encoder uses strides `[1, 2, 2, 1]` across its 4 blocks, downsampling the
frame axis by a total factor of 4 — not all-`1` as the reversed code assumed.

This also exposed a second bug: `context_size_samples`. `model_reversed.py` reads it
off the model (`_model.context_size_samples`) rather than hardcoding it, but the
reconstruction's `VADRNN` never set the attribute at all, so `VAD.forward`'s
`getattr(model, "context_size_samples", 0)` silently fell back to `0` — meaning no
context was ever prepended to the input chunk. Querying the JIT model directly:
`loaded._model.context_size_samples == 64` (16k) / `32` (8k) — this equals
`pad_length`, the `ReflectionPad1d` amount used inside `STFT`. With
`context_size_samples=64`, `hop_length=128`, and encoder strides `[1, 2, 2, 1]`, a
512-sample chunk plus 64 samples of context reduces to exactly **one** frame at the
decoder input (`4 → 4 → 2 → 1 → 1` frames through the encoder stack), matching
`Decoder.forward`'s assumption that its input is `(batch, channels, 1)` — it does
`torch.squeeze(x, -1)` then feeds an `LSTMCell`, which requires the frame dim to
already be 1.

**Fix** (`silero/model_beautified.py`): `SileroVadEncoderBlock` now takes a `stride` argument,
and `VADRNN` builds its 4 encoder blocks with strides `[1, 2, 2, 1]` and sets
`self.context_size_samples = pad_length`.

**Result**: with `load_state_dict` still loading weights, `python -m silero.validate`
showed the full `VAD.forward` output matching the JIT model exactly across a 50-step
streaming sequence (`max err == 0.0`, `mean err == 0.0`).

## Bug 2 — `forward_basis_buffer` was never actually reversed

`STFT.__init__` registered `forward_basis_buffer` as a buffer initialized with
`torch.rand(...)` — a placeholder only used to get the right shape/dtype for
`load_state_dict` to later overwrite. It was never derived from anything, so once
`load_state_dict` is skipped, the STFT breaks (it's the only reason `compute_error_prenet`
passed at all before).

Silero's STFT is a lift of the STFT module long used across NVIDIA's Tacotron2 /
WaveGlow (and derivatives): a fixed (non-learned) real-valued DFT matrix, restricted
to non-redundant frequency bins and windowed, used as a `conv1d` weight:

1. Build the full complex DFT matrix by applying FFT to the identity matrix:
   `fourier_basis = fft(eye(filter_length))` — row `k` is the DFT probe for
   frequency `k`.
2. Keep only the first `cutoff = filter_length // 2 + 1` rows (the rest are the
   redundant complex-conjugate mirror for a real input), and stack real and
   imaginary parts: shape `(2*cutoff, filter_length) = (258, 256)` for
   `filter_length=256`.
3. Reshape to `(258, 1, 256)` to use directly as a `conv1d` weight (out_channels,
   in_channels, kernel_size).
4. Multiply by a periodic (`fftbins=True`) Hann window of length `filter_length`.

Verified numerically against the actual JIT buffer:

```python
forward_basis = fourier_basis  # steps 1-3 above
err_no_window   = ((forward_basis - buf) ** 2).mean()               # 0.1875
err_hann_window = ((forward_basis * hann_window - buf) ** 2).mean() # 0.0
```

No extra scaling (no `sqrt` normalization, no division by `hop_length` or
`filter_length`) is applied — windowing is the only transformation beyond the raw DFT
basis.

**Fix** (`silero/model_beautified.py`): `STFT._make_forward_basis` computes this buffer
directly in `__init__`, ported line-for-line from NVIDIA's Tacotron2 `stft.py` (the
user supplied this original source as reference) — `np.fft.fft` on the identity
matrix, `scipy.signal.get_window(..., fftbins=True)` for the window, and
`librosa.util.pad_center` to center the window inside `filter_length` when
`win_length < filter_length` — instead of registering random values that depend on
`load_state_dict` to become correct, or an earlier hand-rolled `torch.fft`/
`torch.hann_window` version that only matched to float32 rounding noise.

**Result**: with `load_state_dict` disabled entirely (`silero/validate.py`), the
freshly-constructed `forward_basis_buffer` matches the JIT model's buffer **exactly**
(`((ours - jit) ** 2).mean() == 0.0`, bit-for-bit — not just close), and
`compute_error_prenet`'s `mag_err == 0`. The other sub-module errors
(`encoder_err`, `decoder_err`, full-output error) remain nonzero with
`load_state_dict` disabled, as expected — `Conv1d`/`LSTMCell` weights are learned
parameters with no closed-form derivation; only the STFT basis is analytically fixed.

## Takeaway

Decompiled TorchScript `.code` only shows *traced control flow* — module-level
constants that are set once and never appear as an op (`Conv1d.stride`,
`Conv1d.padding`, or an entire buffer computed at `__init__` time like
`forward_basis_buffer`) are invisible to it. Recovering them requires either
inspecting the live submodule attributes directly, or recognizing the buffer as a
known, well-documented DSP construction (here: the standard NVIDIA-style windowed
DFT basis) and re-deriving it analytically rather than assuming it's opaque learned
weight.
