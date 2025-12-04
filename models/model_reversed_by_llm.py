import torch
import torch.nn as nn
import torch.nn.functional as F


class STFT(nn.Module):
    def __init__(self, n_fft, hop_length):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        # Create STFT basis
        self.register_buffer('forward_basis_buffer', self._create_stft_basis(n_fft))
    
    def _create_stft_basis(self, n_fft):
        # Standard STFT basis (Hann window)
        basis = torch.stft(torch.eye(n_fft), n_fft, self.hop_length, return_complex=False)
        return basis.unsqueeze(1)
    
    def forward(self, x):
        # x: (batch, samples)
        return torch.stft(x, self.n_fft, self.hop_length, return_complex=True)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.reparam_conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
    
    def forward(self, x):
        return F.relu(self.reparam_conv(x))


class Encoder(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBlock(input_channels, 128, 3),
            ConvBlock(128, 64, 3),
            ConvBlock(64, 64, 3),
            ConvBlock(64, 128, 3),
        )
    
    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(128, 128, batch_first=True, bidirectional=False)
        self.decoder = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Conv1d(128, 1, 1),
        )
    
    def forward(self, x, state):
        # x: (batch, channels, time_steps)
        # Reshape for RNN: (batch, time_steps, channels)
        x_t = x.transpose(1, 2)
        rnn_out, new_state = self.rnn(x_t, state)
        # Reshape back: (batch, channels, time_steps)
        rnn_out = rnn_out.transpose(1, 2)
        out = self.decoder(rnn_out)
        return out, new_state


class VADRNNJIT(nn.Module):
    """Single VAD model (16k or 8k)"""
    def __init__(self, input_channels):
        super().__init__()
        self.encoder = Encoder(input_channels)
        self.decoder = Decoder()
        self.context_size_samples = 4096 if input_channels == 129 else 2048
    
    def forward(self, x, state):
        # x: (batch, samples)
        # Reshape to (batch, 1, samples) for STFT
        encoded = self.encoder(x.unsqueeze(1) if x.dim() == 2 else x)
        out, new_state = self.decoder(encoded, state)
        return out, new_state


class VADRNNJITMerge(nn.Module):
    """Wrapper that handles both 8k and 16k models"""
    def __init__(self):
        super().__init__()
        self._model = VADRNNJIT(input_channels=129)  # 16k model
        self._model_8k = VADRNNJIT(input_channels=65)  # 8k model
        
        # State management
        self.register_buffer('_state', torch.zeros(1, 1, 128))
        self.register_buffer('_context', torch.zeros(1, 4096))
        self._last_sr = 0
        self._last_batch_size = 0
    
    def _validate_input(self, x, sr):
        """Validate input shape and sample rate"""
        if sr not in [8000, 16000]:
            raise ValueError(f"Sample rate must be 8000 or 16000, got {sr}")
        return x, sr
    
    def reset_states(self):
        """Reset internal state"""
        self._state.zero_()
        self._context.zero_()
        self._last_sr = 0
        self._last_batch_size = 0
    
    def forward(self, x, sr):
        """
        Args:
            x: Audio tensor (batch, samples)
            sr: Sample rate (8000 or 16000)
        
        Returns:
            output: VAD probability (batch, 1, samples)
        """
        x0, sr0 = self._validate_input(x, sr)
        
        # Determine expected number of samples
        num_samples = 512 if sr0 == 16000 else 256
        
        if x0.shape[-1] != num_samples:
            raise ValueError(
                f"Provided number of samples is {x0.shape[-1]} "
                f"(Supported values: 256 for 8000 sample rate, 512 for 16000)"
            )
        
        batch_size = x0.shape[0]
        
        # Select model based on sample rate
        model = self._model if sr0 == 16000 else self._model_8k
        context_size = model.context_size_samples
        
        # Reset if sample rate or batch size changed
        if self._last_sr != 0 and self._last_sr != sr0:
            self.reset_states()
        if self._last_batch_size != 0 and self._last_batch_size != batch_size:
            self.reset_states()
        
        # Initialize context if needed
        if self._context.shape[1] == 0:
            self._context = torch.zeros(batch_size, context_size, device=x0.device)
        
        # Move context to same device as input
        context = self._context.to(x0.device)
        
        # Concatenate context with current input
        x1 = torch.cat([context, x0], dim=-1)
        
        # Forward through selected model
        if self._state.shape[0] != batch_size:
            self._state = torch.zeros(batch_size, 1, 128, device=x0.device)
        
        out, self._state = model(x1, self._state)
        
        # Update context for next iteration
        self._context = x1[:, -context_size:].detach()
        
        # Store metadata
        self._last_sr = sr0
        self._last_batch_size = batch_size
        
        return out.squeeze(1)  # Remove channel dimension


if __name__ == "__main__":
    # Test the module
    model = VADRNNJITMerge()
    
    # Test with 16k audio
    x_16k = torch.randn(1, 512)
    out_16k = model(x_16k, 16000)
    print(f"16k output shape: {out_16k.shape}")
    
    # Test with 8k audio
    x_8k = torch.randn(1, 256)
    out_8k = model(x_8k, 8000)
    print(f"8k output shape: {out_8k.shape}")