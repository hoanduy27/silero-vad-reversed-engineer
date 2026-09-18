import torch
from silero_vad.utils_vad import init_jit_model
from silero.model_beautified import VAD

SAMPLE_RATE = 16000

model_reversed = VAD()

model = torch.jit.load(
    "/home/duy/miniconda3/envs/bnlu/lib/python3.13/site-packages/silero_vad/data/silero_vad.jit",
)

model_reversed._model = model._model

model.eval()
model_reversed.eval()


for i in range(10):

    t = torch.rand((1, 512))

    out_jit = model(t, sr=SAMPLE_RATE)
    out_rv = model_reversed(t, sr=SAMPLE_RATE)
    
    

    error = torch.abs(out_jit - out_rv).max()
    print("Max error:", error.item())
    
    state = model_reversed._state
    state_jit = model._state
    
    error_state = torch.abs(state - state_jit).max()
    
    print("Max state error:", error_state.item())