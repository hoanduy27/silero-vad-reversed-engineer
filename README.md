# SileroVAD-reversed-engineer

Effort to reversed engineer Silero VAD for deeper custom training and optimization.

- Why?
  - Silero VAD is production-ready due to it's being lightweight.
  - However, it tricky scenario, like **human background noise** or **in the middle of the sentence**, this model can be failed to generalize.
  - Fortunately, SileroVAD's code is encoded in TorchScript (think of it as minified javascript). Which means, we can still reverse the code of its architecture with minimal guess. *We still need to guess and write a training script for SileroVAD however.*

- This project hopes to open a new door for customization SileroVAD for production needs.

- What we want to reverse:
  - The architecture (easier due to availability of the minified TorchScript code), then
  - The training strategy (should takes more effort).


# Writeups
- Our exploration is stored in [This notebook](explore.ipynb) (it's not well structured yet). 
- The reversed code is stored in [silero module](silero), which reflects the progress of our exploration.
  - `model_rv.py`: The code extracted from TorchScript.
  - `model.py`: The unminified code for `model_rv.py`. We use Copilot to aid with this process.
  - `validate.py`: To check whether the reversed code is correct or not.
  - `trainer.py`: The training script for SileroVAD (TBU).

# Current progress:
[-]: In progress, [x]: Done, [ ]: Plan to do


- [-] Architectural reversal: Working on 16khz model
  - [x] Numerical errors for each building blocks now are **zero** for random tensor sequences.
  - [-] Numerical errors for full data flow still is **non-zero**. We need further investigation.

- [ ] Training reversal
