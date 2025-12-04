import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional

# Extracted forward methods from Silero VAD JIT model
class VAD(nn.Module):
  def __init__(self):
    self.model = ...
    
  def reset_states(self):
    self._state = torch.zeros([0])
    self._context = torch.zeros([0])
    self._last_sr = 0
    self._last_batch_size = 0
    return None
  
  def _validate_input(self,
    x: torch.Tensor,
    sr: int) -> Tuple[torch.Tensor, int]:
    
    _0 = "Too many dimensions for input audio chunk {}"
    _1 = "Supported sampling rates: {} (or multiply of 16000)"
    if len(x.size()) == 1:
      x = torch.unsqueeze(x, 0)
    
    if len(x.size()) > 2:
      raise ValueError(f"Too many dimensions for input audio chunk {x.size()}")
    
    # if sr != 16000:
    #   _3 = torch.eq(torch.remainder(sr, 16000), 0)
    #   _2 = _3
    # else:
    #   _2 = False
      
    if sr % 16000:
      step = torch.floordiv(sr, 16000)
      x4 = torch.slice(torch.slice(x), 1, None, None, step)
      sr1, x3 = 16000, x4
    else:
      sr1, x3 = sr, x
    sample_rates = self.sample_rates
    _4 = torch.__not__(torch.__contains__(sample_rates, sr1))
    if _4:
      sample_rates0 = self.sample_rates
      raise ValueError(f"Supported sampling rates: {sample_rates0} (or multiply of 16000)")
      
    else:
      pass
    _5 = torch.gt(torch.div(sr1, (torch.size(x3))[1]), 31.25)
    if _5:
      raise ValueError("Input audio chunk is too short")
    else:
      pass
    return (x3, sr1)

  def forward(self,
      x: Tensor,
      sr: int) -> Tensor:
    _0 = "Provided number of samples is {} (Supported values: 256 for 8000 sample rate, 512 for 16000)"
    _1 = torch.empty(0)
    x0, sr0, = (self)._validate_input(x, sr, )
    if torch.eq(sr0, 16000):
      num_samples = 512
    else:
      num_samples = 256
    _2 = torch.ne((torch.size(x0))[-1], num_samples)
    if _2:
      _3 = torch.format(_0, (torch.size(x0))[-1])
      raise ValueError(_3)
    else:
      pass
    batch_size = (torch.size(x0))[0]
    if torch.eq(sr0, 16000):
      _model = self._model
      context_size = _model.context_size_samples
    else:
      _model_8k = self._model_8k
      context_size = _model_8k.context_size_samples
    _last_sr = self._last_sr
    if bool(_last_sr):
      _last_sr0 = self._last_sr
      _4 = torch.ne(_last_sr0, sr0)
    else:
      _4 = False
    if _4:
      _5 = (self).reset_states()
    else:
      pass
    _last_batch_size = self._last_batch_size
    if bool(_last_batch_size):
      _last_batch_size0 = self._last_batch_size
      _7 = torch.ne(_last_batch_size0, batch_size)
      _6 = _7
    else:
      _6 = False
    if _6:
      _8 = (self).reset_states()
    else:
      pass
    _context = self._context
    _9 = torch.__not__(bool(torch.len(_context)))
    if _9:
      _10 = torch.zeros([batch_size, context_size])
      self._context = _10
    else:
      pass
    _context0 = self._context
    _11 = torch.to(_context0, x0.device)
    x1 = torch.cat([_11, x0], 1)
    if torch.eq(sr0, 16000):
      _model0 = self._model
      _state = self._state
      out0, _12, = (_model0).forward(x1, _state, )
      self._state = _12
      out = out0
    else:
      if torch.eq(sr0, 8000):
        _model_8k0 = self._model_8k
        _state0 = self._state
        _13 = (_model_8k0).forward(x1, _state0, )
        out2, _14, = _13
        self._state = _14
        out1 = out2
      else:
        out1 = _1
        raise ValueError("")
      out = out1
    _15 = torch.slice(x1, -1, torch.neg(context_size))
    self._context = _15
    self._last_sr = sr0
    self._last_batch_size = batch_size
    return out

class STFT(nn.Module):
  def __init__(self, hop_length=128, filter_length=256, pad_length=64):
    self.padding = nn.ReflectionPad1d(padding=(0, pad_length))
    self.hop_length = hop_length
    self.filter_length = filter_length
    
  def forward(self, input_data):
    input_data = torch.unsqueeze(self.padding(input_data, ), 1)
    
    
    
    forward_basis_buffer = self.forward_basis_buffer
    hop_length = self.hop_length
    
    forward_transform = torch.conv1d(
        input_data, 
        weight=forward_basis_buffer, 
        bias=None, 
        stride=[hop_length], 
        padding=[0]
    )
    
    # filter_length = self.filter_length
    
    # _0 = torch.add(torch.div(filter_length, 2), 1)
    
    cutoff = int((self.filter_length / 2) + 1)
    
    # cutoff = int(_0)
    
    _1 = torch.slice(torch.slice(forward_transform), 1, None, cutoff)
    real_part = torch.to(torch.slice(_1, 2), 6)
    
    _2 = torch.slice(torch.slice(forward_transform), 1, cutoff)
    imag_part = torch.to(torch.slice(_2, 2), 6)
    
    magnitude = torch.sqrt(
      torch.pow(real_part, 2) + torch.pow(imag_part, 2)
    )
    
    phase = torch.atan2(imag_part, real_part)
    
    return (magnitude, phase)



class SileroVadEncoderBlock(nn.Module):
  """
  RecursiveScriptModule(
    original_name=SileroVadBlock
    (se): RecursiveScriptModule(original_name=Identity)
    (activation): RecursiveScriptModule(original_name=ReLU)
    (reparam_conv): RecursiveScriptModule(original_name=Conv1d)
  )
  """
  def __init__(self, in_channels, out_channels):    
    self.se = nn.Identity()
    self.activation = nn.ReLU()
    self.reparam_conv = nn.Conv1d(
      in_channels=in_channels,
      out_channels=out_channels,
      kernel_size=3,
      padding=1,
    )
    
  def forward(self,
    x: Tensor) -> Tensor:
    
    activation = self.activation
    se = self.se
    reparam_conv = self.reparam_conv
    _0 = (se).forward((reparam_conv).forward(x, ), )
    return (activation).forward(_0, )
  
class Decoder(nn.Module):
  def __init__(self):
    self.decoder = nn.Sequential(
      nn.Dropout(),
      nn.ReLU(),
      nn.Conv1d(
        in_channels=128,
        out_channels=1,
        kernel_size=1,
        padding=0  
      ),
      nn.Sigmoid()
    )
    
    
    self.rnn = nn.LSTMCell(
      input_size=128,
      hidden_size=128
    )
    
    
  def forward(self,
    x: Tensor,
    state: Tensor=None) -> Tuple[Tensor, Tensor]:
    
    x0 = torch.squeeze(x, -1)
    if bool(torch.len(state)):
      rnn = self.rnn
      _0 = (torch.select(state, 0, 0), torch.select(state, 0, 1))
      h0, c0, = (rnn).forward(x0, _0, )
      h, c = h0, c0
    else:
      rnn0 = self.rnn
      h1, c1, = (rnn0).forward(x0, None, )
      h, c = h1, c1
    x1 = torch.to(torch.unsqueeze(h, -1), 6)
    state0 = torch.stack([h, c])
    decoder = self.decoder
    x2 = (decoder).forward(x1, )
    return (x2, state0)
  

class VADRNN(nn.Module):
  def __init__(self, hop_length=128, filter_length=256, pad_length=64):
    self.stft = STFT(hop_length=hop_length, filter_length=filter_length, pad_length=pad_length)
    
    
    self.encoder = nn.Sequential(
      SileroVadEncoderBlock(in_channels=129, out_channels=128),
      SileroVadEncoderBlock(in_channels=128, out_channels=64),
      SileroVadEncoderBlock(in_channels=64, out_channels=64),
      SileroVadEncoderBlock(in_channels=64, out_channels=128),
    )
    self.decoder = Decoder()
    
    
  
  def forward(self,
    x: Tensor,
    state: Tensor=None) -> Tuple[Tensor, Tensor]:
    # 512 -> [129] [129] 
    x0 = self.stft(x, )
    
    x1 = self.encoder(x0, )
    
    x2, state0 = self.decoder(x1, state, )
    
    out = torch.unsqueeze(torch.mean(torch.squeeze(x2, 1), [1]), 1)
    
    return (out, state0)
  