import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional
import torch.nn.functional as F
import numpy as np
from scipy.signal import get_window
from librosa.util import pad_center

class VAD(nn.Module):
    """
    Wrapper that manages state/context around two internal models:
    - self._model for 16 kHz
    - self._model_8k for 8 kHz
    Expects these submodels to expose `.context_size_samples` and a `.forward(x, state)` signature.
    """

    # Class-body type annotations -- TorchScript's scripting needs Optional[Tensor]
    # declared this way (not inline in __init__) to type-narrow `is None` checks
    # in forward() correctly.
    _state: Optional[Tensor]
    _context: Optional[Tensor]

    def __init__(self, sample_rates=(8000, 16000)):
        super().__init__()
        # caller should set self._model and self._model_8k after construction
        self._model = None
        self._model_8k = None
        self.sample_rates = tuple(sample_rates)

        # streaming state
        self.reset_states()

    def reset_states(self):
        self._state = None
        self._context = None
        self._last_sr = 0
        self._last_batch_size = 0

    def _validate_input(self, x: Tensor, sr: int) -> Tuple[Tensor, int]:
        """
        Normalize 1D/2D input to shape (batch, samples).
        If sr is a multiple of 16000 (e.g. 32000) we downsample by integer step.
        Checks that resulting sampling rate is supported and that the chunk is not too short.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)  # make batch dim

        if x.dim() > 2:
            raise ValueError(f"Too many dimensions for input audio chunk {x.size()}")

        # If sampling rate is a multiple of 16000 *other than* 16000 itself
        # (e.g. 32000), downsample to 16000 by taking every `step`-th sample.
        # sr == 8000 (or any other non-multiple) falls through unchanged, to
        # be checked against self.sample_rates below.
        if sr != 16000 and sr % 16000 == 0:
            step = sr // 16000
            x = x[..., ::step]
            sr_effective = 16000
        else:
            sr_effective = sr

        if sr_effective not in self.sample_rates:
            raise ValueError(f"Supported sampling rates: {self.sample_rates} (or multiple of 16000)")

        # Check chunk length isn't absurdly short (keeps the original guard logic)
        num_samples = x.size(1)
        if (sr_effective / float(num_samples)) > 31.25:
            raise ValueError("Input audio chunk is too short")

        return x, sr_effective

    def forward(self, x: Tensor, sr: int) -> Tensor:
        """
        x: (batch, samples) or (samples,)
        sr: sampling rate (8k, 16k or multiple of 16k)
        Returns model output for the given chunk and updates internal streaming state.
        """
        x, sr = self._validate_input(x, sr)

        num_samples = 512 if sr == 16000 else 256
        if x.size(-1) != num_samples:
            raise ValueError(f"Provided number of samples is {x.size(-1)} (Supported values: 256 for 8000 sample rate, 512 for 16000)")

        batch_size = x.size(0)

        # context size based on sampling rate -- accessed directly per-branch
        # (self._model.forward(...) / self._model_8k.forward(...), likewise
        # below) rather than through a shared local variable: TorchScript
        # can't unify a local that might statically be either submodule into
        # one callable, even though both are structurally VADRNN.
        if sr == 16000:
            context_size = self._model.context_size_samples
        else:
            context_size = self._model_8k.context_size_samples

        # Reset streaming states if sampling rate or batch size changed
        if self._last_sr and self._last_sr != sr:
            self.reset_states()
        if self._last_batch_size and self._last_batch_size != batch_size:
            self.reset_states()

        # Ensure context tensor exists and is on the same device as input.
        # Bound to a local first -- TorchScript doesn't reliably type-narrow
        # `self.attr is None` followed by `self.attr.method()` in the same
        # expression the way it does for a local variable.
        context = self._context
        if context is None or context.size(0) != batch_size:
            # context is (batch, context_size)
            context = torch.zeros(batch_size, context_size, device=x.device, dtype=x.dtype)
        else:
            context = context.to(x.device)
        self._context = context

        # Concatenate context and new chunk along sample dimension
        x_with_context = torch.cat([context, x], dim=1)

        # Call the selected model's forward and update hidden state.
        # self._model_8k, when present, is the original (pre-scripted, never
        # fine-tuned) checkpoint's 8kHz submodule reused as-is -- its `state`
        # param is a plain Tensor with an empty-tensor sentinel for "none yet"
        # (that convention, not Optional[Tensor], is what a module already
        # compiled by torch.jit.script elsewhere requires), unlike our own
        # VADRNN.forward's Optional[Tensor].
        state = self._state
        if sr == 16000:
            out, new_state = self._model.forward(x_with_context, state)
        elif sr == 8000:
            state_8k = state if state is not None else torch.empty(0)
            out, new_state = self._model_8k.forward(x_with_context, state_8k)
        else:
            # unreachable due to validation above
            raise ValueError(f"Unsupported sampling rate {sr}")

        self._state = new_state

        # Keep trailing `context_size` samples for next call
        if context_size > 0:
            self._context = x_with_context[:, -context_size:].detach()
        else:
            self._context = torch.zeros(batch_size, 0, device=x.device, dtype=x.dtype)

        self._last_sr = sr
        self._last_batch_size = batch_size

        return out


class STFT(nn.Module):
    """
    Simple STFT-style forward using a precomputed forward_basis_buffer.
    forward_basis_buffer is expected to be a nn.Parameter or buffer shaped
    (out_channels, in_channels, kernel_width) appropriate for conv1d.
    The convolution is expected to produce a tensor where the first half along
    channel dimension is real-part bins and the second half is imaginary-part bins.
    """

    def __init__(self, hop_length=128, filter_length=256, pad_length=64, win_length=None, window='hann'):
        super().__init__()
        self.padding = nn.ReflectionPad1d(padding=(0, pad_length))
        self.hop_length = hop_length
        self.filter_length = filter_length

        self.register_buffer(
          name="forward_basis_buffer",
          tensor=self._make_forward_basis(filter_length, win_length or filter_length, window)
        )

    @staticmethod
    def _make_forward_basis(filter_length: int, win_length: int, window: str) -> Tensor:
        # Ported from NVIDIA's Tacotron2 STFT (stft.py), which Silero's STFT is
        # itself adapted from: a fixed (non-trainable) real DFT basis, restricted
        # to non-redundant frequency bins, windowed, and shaped as a conv1d weight.
        cutoff = int(filter_length / 2 + 1)
        fourier_basis = np.fft.fft(np.eye(filter_length))
        fourier_basis = np.vstack(
            [np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])]
        )
        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])

        if window is not None:
            assert filter_length >= win_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, size=filter_length)
            fft_window = torch.from_numpy(fft_window).float()
            forward_basis *= fft_window

        return forward_basis

    def forward(self, input_data: Tensor) -> Tensor:
        """
        input_data: (batch, samples)
        returns: magnitude, shape (batch, freq_bins, frames)
        """
        # pad and add channel dim for conv1d: shape (batch, 1, samples_padded)
        x = input_data
        x = self.padding(x)
        x = x.unsqueeze(1)

        # conv1d expects weight shape (out_channels, in_channels, kernel_size)
        conv = F.conv1d(x, self.forward_basis_buffer, bias=None, stride=self.hop_length, padding=0)

        cutoff = (self.filter_length // 2) + 1  # number of frequency bins
        real = conv[:, :cutoff, :]
        imag = conv[:, cutoff:cutoff * 2, :]

        magnitude = torch.sqrt(real.pow(2) + imag.pow(2) + 1e-12)
        return magnitude


class SileroVadEncoderBlock(nn.Module):
    """
    Single encoder block: Conv1d -> (possibly SE) -> ReLU
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.se = nn.Identity()
        self.activation = nn.ReLU()
        self.reparam_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=stride)

    def forward(self, x: Tensor) -> Tensor:
        x = self.reparam_conv(x)
        x = self.se(x)
        x = self.activation(x)
        return x


class BranchedEncoderBlock(nn.Module):
    """
    RepVGG-style re-branched version of SileroVadEncoderBlock, for fine-tuning.

    We do NOT attempt to recover the original (pre-fusion) branch weights --
    that reconstruction is lossy/non-invertible, see validate_rep_vgg.py. Instead
    we keep the deployed 3x3 conv exactly as-is and bolt on a *new*, zero-initialized
    1x1 branch (and, optionally, a zero-initialized identity/BN branch when
    in_channels == out_channels and stride == 1). Because both new branches start
    at zero contribution, `BranchedEncoderBlock.from_reparam(block)` is numerically
    identical to the original block until training moves the new branches away
    from zero. `reparameterize()` performs the inverse: fold the branches back into
    a single Conv1d, algebraically equivalent to RepVGG's deploy-time fusion.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, use_identity: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.se = nn.Identity()
        self.activation = nn.ReLU()

        self.conv3x1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)

        self.conv1x1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride)
        nn.init.zeros_(self.conv1x1.weight)
        nn.init.zeros_(self.conv1x1.bias)

        self.has_identity = use_identity and (in_channels == out_channels) and (stride == 1)
        if self.has_identity:
            # BatchNorm1d applied directly to the block input (RepVGG's identity
            # branch). Zero-initialized affine weight => zero contribution at init,
            # regardless of running statistics.
            self.identity_bn = nn.BatchNorm1d(in_channels)
            nn.init.zeros_(self.identity_bn.weight)
            nn.init.zeros_(self.identity_bn.bias)
        else:
            self.identity_bn = None

    @classmethod
    def from_reparam(cls, block: "SileroVadEncoderBlock", use_identity: bool = True) -> "BranchedEncoderBlock":
        """Build a branched block whose forward() output matches `block` exactly."""
        stride = block.reparam_conv.stride[0]
        in_channels = block.reparam_conv.in_channels
        out_channels = block.reparam_conv.out_channels

        new_block = cls(in_channels, out_channels, stride=stride, use_identity=use_identity)
        new_block.conv3x1.weight.data.copy_(block.reparam_conv.weight.data)
        new_block.conv3x1.bias.data.copy_(block.reparam_conv.bias.data)
        return new_block

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv3x1(x) + self.conv1x1(x)
        if self.has_identity:
            out = out + self.identity_bn(x)
        out = self.se(out)
        out = self.activation(out)
        return out

    @staticmethod
    def _pad_1x1_to_3x1(weight_1x1: Tensor) -> Tensor:
        # (out, in, 1) -> (out, in, 3), placing the single tap in the center
        return F.pad(weight_1x1, [1, 1])

    def _identity_to_3x1(self) -> Tuple[Tensor, Tensor]:
        # A BatchNorm1d applied to x is equivalent, per output channel c, to a
        # 1x1 conv with weight gamma_c / sqrt(var_c + eps) on the diagonal (c, c)
        # and bias beta_c - mean_c * gamma_c / sqrt(var_c + eps). Padded to 3x1
        # like the 1x1 branch, this can be summed directly into the fused kernel.
        bn = self.identity_bn
        std = torch.sqrt(bn.running_var + bn.eps)
        scale = bn.weight / std
        bias = bn.bias - bn.running_mean * scale

        weight = torch.zeros(self.out_channels, self.in_channels, 1, device=scale.device, dtype=scale.dtype)
        diag_idx = torch.arange(self.in_channels, device=scale.device)
        weight[diag_idx, diag_idx, 0] = scale
        return self._pad_1x1_to_3x1(weight), bias

    def reparameterize(self) -> nn.Conv1d:
        """Fold conv3x1 + conv1x1 (+ identity_bn) into a single deployable Conv1d."""
        fused_weight = self.conv3x1.weight.data.clone()
        fused_bias = self.conv3x1.bias.data.clone()

        fused_weight = fused_weight + self._pad_1x1_to_3x1(self.conv1x1.weight.data)
        fused_bias = fused_bias + self.conv1x1.bias.data

        if self.has_identity:
            id_weight, id_bias = self._identity_to_3x1()
            fused_weight = fused_weight + id_weight
            fused_bias = fused_bias + id_bias

        fused_conv = nn.Conv1d(self.in_channels, self.out_channels, kernel_size=3, padding=1, stride=self.stride)
        fused_conv.weight.data.copy_(fused_weight)
        fused_conv.bias.data.copy_(fused_bias)
        return fused_conv

    def to_reparam_block(self) -> "SileroVadEncoderBlock":
        """Convenience: return a plain SileroVadEncoderBlock with the fused conv."""
        block = SileroVadEncoderBlock(self.in_channels, self.out_channels, stride=self.stride)
        block.reparam_conv = self.reparameterize()
        return block


class Decoder(nn.Module):
    """
    Decoder that contains an LSTMCell-based recurrent stage and a small conv head.
    - Expects input x shaped (batch, channels, 1) or (batch, channels) where channels==128.
    - state is expected to be a tensor shaped (2, batch, hidden) or None.
    - returns (output, state) where output has shape (batch, 1, frames) or (batch, 1, 1) depending on input.
    """

    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Dropout(),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.rnn = nn.LSTMCell(input_size=128, hidden_size=128)

    def forward(self, x: Tensor, state: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        # x expected shape: (batch, 128, 1) or (batch, 128)
        if x.dim() == 3 and x.size(2) == 1:
            x_in = x.squeeze(-1)  # (batch, 128)
        else:
            x_in = x

        batch = x_in.size(0)

        if state is not None:
            # expected state shape: (2, batch, hidden) where state[0] = h, state[1] = c
            h0 = state[0]
            c0 = state[1]
            h, c = self.rnn(x_in, (h0, c0))
        else:
            # Initialize hidden/cell as zeros if not provided
            h = torch.zeros(batch, self.rnn.hidden_size, device=x_in.device, dtype=x_in.dtype)
            c = torch.zeros(batch, self.rnn.hidden_size, device=x_in.device, dtype=x_in.dtype)
            h, c = self.rnn(x_in, (h, c))

        # prepare state for return: shape (2, batch, hidden)
        state_out = torch.stack([h, c], dim=0)

        # pass through conv decoder: need shape (batch, channels, time)
        x_feat = h.unsqueeze(-1)  # (batch, 128, 1)
        out = self.decoder(x_feat)  # (batch, 1, 1)

        return out, state_out


class VADRNN(nn.Module):
    """
    Full VAD RNN: STFT -> encoder blocks -> decoder RNN
    """

    def __init__(self, hop_length=128, filter_length=256, pad_length=64):
        super().__init__()
        self.stft = STFT(hop_length=hop_length, filter_length=filter_length, pad_length=pad_length)

        # encoder chain expects freq bins = filter_length // 2 + 1 = 129 (for filter_length=256)
        # strides [1, 2, 2, 1] downsample the frame axis so that context_size_samples
        # worth of lookback collapses back to exactly one frame at the decoder input
        self.encoder = nn.Sequential(
            SileroVadEncoderBlock(in_channels=(filter_length // 2 + 1), out_channels=128, stride=1),
            SileroVadEncoderBlock(in_channels=128, out_channels=64, stride=2),
            SileroVadEncoderBlock(in_channels=64, out_channels=64, stride=2),
            SileroVadEncoderBlock(in_channels=64, out_channels=128, stride=1),
        )
        self.decoder = Decoder()

        # number of samples carried over from the previous chunk as context;
        # matches the reflect-padding length used by the STFT
        self.context_size_samples = pad_length

    def forward(self, x: Tensor, state: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        # x: (batch, samples) or (samples,)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # STFT returns (magnitude, phase)
        magnitude = self.stft(x)

        # encoder expects (batch, freq_bins, frames)
        encoded = self.encoder(magnitude)

        # decoder expects (batch, channels, 1) per time-step in original model.
        # If there are multiple frames, here we will average over frames to produce a single decision per chunk.
        # squeeze frames if singleton; otherwise take mean over frames before feeding decoder.
        if encoded.dim() == 3:
            # encoded: (batch, channels, frames)
            # collapse frames by mean for a single-step decoder
            encoded_mean = encoded.mean(dim=2, keepdim=True)
        else:
            encoded_mean = encoded

        out, new_state = self.decoder(encoded_mean, state)

        # out: (batch, 1, 1) -> produce (batch, 1, 1) probability. Keep consistent shape.
        # For compatibility with caller that expected a (batch, 1, 1) or (batch, 1) shape:
        out_mean = out.mean(dim=-1, keepdim=True)  # (batch, 1, 1)

        return out_mean, new_state

    def forward_embed(self, x: Tensor, return_individuals: bool = False) -> Tensor:
        """
        Encoder-only forward pass for utterance-level embeddings (e.g. speaker
        similarity), bypassing the decoder RNN entirely.

        x: (batch, samples) or (samples,) — a full utterance, not a fixed-size
        streaming chunk, since the encoder is purely convolutional and has no
        recurrent state to carry across chunk boundaries.

        return_individuals: if True, skip the pooling and return the encoder's
        per-frame output as-is — one embedding per encoder time step, instead
        of a single utterance-level summary.

        Returns:
        - return_individuals=False: (batch, 128) fixed-size embedding,
          obtained by mean-pooling the encoder's per-frame output
          (batch, 128, frames) over time.
        - return_individuals=True: (batch, frames, 128) per-frame embeddings.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        magnitude = self.stft(x)
        encoded = self.encoder(magnitude)  # (batch, 128, frames)

        if return_individuals:
            return encoded.transpose(1, 2)  # (batch, frames, 128)

        embedding = encoded.mean(dim=2)  # (batch, 128)

        return embedding

    def forward_sequence(self, x: Tensor) -> Tensor:
        """
        Full-utterance training forward: reproduces VAD.forward's per-chunk
        computation exactly (each 512-sample chunk gets context_size_samples=64
        raw-sample look-back prepended, LSTM state carried across chunks) --
        unlike naively running STFT + encoder once over the whole continuous
        signal, which does NOT reproduce the same per-frame decisions: each
        streaming chunk's STFT window is anchored context_size_samples (64 =
        half a hop_length) earlier than a global, whole-signal frame grid would
        place it, and its trailing reflection-padding reflects that chunk's own
        tail rather than the signal's true continuation.

        Unlike a literal per-chunk port, though, the STFT+encoder half of this
        (unlike the decoder) has no recurrence -- each chunk's window is fully
        determined by raw audio, independent of any other chunk's decision --
        so all chunks' windows are built and run through STFT+encoder as ONE
        batched call via `unfold` instead of a Python loop over tiny per-chunk
        conv calls, which is what actually keeps the GPU busy; only the
        decoder's LSTMCell, which is inherently sequential, still loops.

        x: (batch, samples) or (samples,)
        Returns: (batch, num_chunks) speech probabilities, one per 512-sample
        chunk (matching the streaming `num_samples=512` convention for 16kHz).
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        chunk_size = 512
        batch = x.size(0)
        num_chunks = x.size(1) // chunk_size

        leading_zeros = torch.zeros(batch, self.context_size_samples, device=x.device, dtype=x.dtype)
        padded = torch.cat([leading_zeros, x[:, :num_chunks * chunk_size]], dim=1)
        # windows[:, i, :] == cat([context_for_chunk_i, chunk_i]), exactly what
        # the per-chunk loop would build (context_for_chunk_0 is the leading
        # zeros; context_for_chunk_i>0 is chunk_{i-1}'s trailing context_size_samples).
        windows = padded.unfold(1, self.context_size_samples + chunk_size, chunk_size)  # (batch, num_chunks, ctx+chunk)

        windows_flat = windows.reshape(batch * num_chunks, -1)
        magnitude = self.stft(windows_flat)
        encoded = self.encoder(magnitude)  # (batch*num_chunks, 128, 1) -- one frame/window
        encoded = encoded.reshape(batch, num_chunks, -1)  # (batch, num_chunks, 128)

        state = None
        outs = []
        for i in range(num_chunks):
            frame = encoded[:, i, :].unsqueeze(-1)  # (batch, 128, 1)
            out, state = self.decoder(frame, state)  # out: (batch, 1, 1)
            outs.append(out.squeeze(-1).squeeze(-1))  # (batch,)

        return torch.stack(outs, dim=1)  # (batch, num_chunks)


