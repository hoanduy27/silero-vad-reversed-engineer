import torch
from silero.model_beautified import VAD, VADRNN
from collections import OrderedDict

state_dict = OrderedDict()

class Validator:
    def __init__(self):
        self.vad = VAD()
        self.vad._model = VADRNN()

        self.loaded = torch.jit.load("assets/silero_vad.jit")

        loaded_state_dict = self.loaded._model.state_dict()
        loaded_state_dict.pop("stft.forward_basis_buffer")
        self.vad._model.load_state_dict(loaded_state_dict, strict=False)

        self.loaded._model.eval()
        self.vad._model.eval()

    def reset_states(self):
        self.loaded.reset_states()
        self.vad.reset_states()

    def compute_error(self):
        t = torch.rand((1, 512))
        sr = 16000
        out_jit = self.loaded(t, sr=sr)
        out_rv = self.vad(t, sr=sr)
        
        print(out_jit)
        print(out_rv)
        
    def compute_error_encoder(self):
        t = torch.rand((1, 129, 1))
        
        out_orig = self.loaded._model.encoder(t)
        # print(out_orig)
        out_reversed = self.vad._model.encoder(t)
        # print(out_reversed)
        
        err = ((out_orig - out_reversed)**2).mean()
        return err
        
    def compute_error_decoder_single_chunk(self):
        t = torch.rand((1, 128))
        
        out_orig, state_out_orig = self.loaded._model.decoder(t)
        out_reversed, state_out_reversed = self.vad._model.decoder(t)
        
        err_out = ((out_orig - out_reversed)**2).mean()
        err_state = ((state_out_orig - state_out_reversed)**2).mean()
        
        return err_out, err_state
    
    def compute_error_decoder_sequence(self, sequence_length=100):
        err_outs = []
        err_states = []
        
        state_orig = None
        state_reversed = None
        
        for i in range(sequence_length):
            t = torch.rand((1, 128))
            out_orig, state_orig = self.loaded._model.decoder(t, state_orig)
            out_reversed, state_reversed = self.vad._model.decoder(t, state_reversed)
            
            err_out = ((out_orig - out_reversed)**2).mean()
            err_state = ((state_orig - state_reversed)**2).mean()
            
            
            err_outs.append(err_out)
            err_states.append(err_state)
            
        err_outs = torch.stack(err_outs)    
        err_states = torch.stack(err_states)
        
        return err_outs.mean(), err_states.mean()
    
    def compute_error_prenet(self):
        t = torch.rand((1, 4096))
        orig = self.loaded._model.stft(t)
        reversed = self.vad._model.stft(t)
        
        err = ((orig - reversed)**2).mean()
        
        return err
        
v = Validator()
# v.compute_error()
encoder_err = v.compute_error_encoder()
print(f"{encoder_err=}")

v.reset_states()
decoder_err, state_err = v.compute_error_decoder_single_chunk()
print(f"{decoder_err=}")
print(f"{state_err=}")

v.reset_states()
decoder_err_sequence, state_err_sequence = v.compute_error_decoder_sequence()
print(f"{decoder_err_sequence=}")
print(f"{state_err_sequence=}")


v.reset_states()
mag_err = v.compute_error_prenet()
print(f"{mag_err=}")


v.reset_states()
v.compute_error()