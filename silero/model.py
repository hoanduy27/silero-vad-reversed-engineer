import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional
import torch.nn.functional as F


class VAD(nn.Module):
    """
    Wrapper that manages state/context around two internal models:
    - self._model for 16 kHz
    - self._model_8k for 8 kHz
    Expects these submodels to expose `.context_size_samples` and a `.forward(x, state)` signature.
    """

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

        # If sampling rate is multiple of 16000, downsample to 16000 by taking every `step`-th sample
        if sr % 16000 != 0:
            step = sr // 16000
            if step <= 0:
                raise ValueError(f"Unsupported sampling rate: {sr}")
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

        # choose model and context size based on sampling rate
        if sr == 16000:
            model = self._model
            if model is None:
                raise RuntimeError("self._model (16k) is not set")
            context_size = getattr(model, "context_size_samples", 0)
        else:
            model = self._model_8k
            if model is None:
                raise RuntimeError("self._model_8k (8k) is not set")
            context_size = getattr(model, "context_size_samples", 0)

        # Reset streaming states if sampling rate or batch size changed
        if self._last_sr and self._last_sr != sr:
            self.reset_states()
        if self._last_batch_size and self._last_batch_size != batch_size:
            self.reset_states()

        # Ensure context tensor exists and is on the same device as input
        if self._context is None or self._context.size(0) != batch_size:
            # context is (batch, context_size)
            self._context = torch.zeros(batch_size, context_size, device=x.device, dtype=x.dtype)
        else:
            self._context = self._context.to(x.device)

        # Concatenate context and new chunk along sample dimension
        x_with_context = torch.cat([self._context, x], dim=1)

        # Call the selected model's forward and update hidden state
        if sr == 16000:
            out, new_state = model.forward(x_with_context, self._state)
        elif sr == 8000:
            out, new_state = model.forward(x_with_context, self._state)
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

    def __init__(self, hop_length=128, filter_length=256, pad_length=64):
        super().__init__()
        self.padding = nn.ReflectionPad1d(padding=(0, pad_length))
        self.hop_length = hop_length
        self.filter_length = filter_length

        # forward_basis_buffer should be provided by the caller (as buffer/param)
        
        self.register_buffer(
          name="forward_basis_buffer",
          tensor=torch.rand((filter_length + 2, 1, filter_length))
        )

    def forward(self, input_data: Tensor) -> Tuple[Tensor, Tensor]:
        """
        input_data: (batch, samples)
        returns: (magnitude, phase) where both are (batch, freq_bins, frames)
        """
        # pad and add channel dim for conv1d: shape (batch, 1, samples_padded)
        x = input_data
        x = self.padding(x)
        x = x.unsqueeze(1)

        if self.forward_basis_buffer is None:
            raise RuntimeError("forward_basis_buffer is not set on STFT")

        # conv1d expects weight shape (out_channels, in_channels, kernel_size)
        conv = F.conv1d(x, self.forward_basis_buffer, bias=None, stride=self.hop_length, padding=0)

        cutoff = (self.filter_length // 2) + 1  # number of frequency bins
        # assume conv produces channels = 2 * cutoff (real then imag)
        if conv.size(1) < 2 * cutoff:
            raise RuntimeError("forward transform produced too few channels for split into real/imag parts")

        real = conv[:, :cutoff, :]
        imag = conv[:, cutoff:cutoff*2, :]

        magnitude = torch.sqrt(real.pow(2) + imag.pow(2) + 1e-12)
        phase = torch.atan2(imag, real)

        return magnitude


class SileroVadEncoderBlock(nn.Module):
    """
    Single encoder block: Conv1d -> (possibly SE) -> ReLU
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.se = nn.Identity()
        self.activation = nn.ReLU()
        self.reparam_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.reparam_conv(x)
        x = self.se(x)
        x = self.activation(x)
        return x


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
        self.encoder = nn.Sequential(
            SileroVadEncoderBlock(in_channels=(filter_length // 2 + 1), out_channels=128),
            SileroVadEncoderBlock(in_channels=128, out_channels=64),
            SileroVadEncoderBlock(in_channels=64, out_channels=64),
            SileroVadEncoderBlock(in_channels=64, out_channels=128),
        )
        self.decoder = Decoder()

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


