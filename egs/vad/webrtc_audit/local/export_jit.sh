#!/usr/bin/env bash
# Export the fine-tuned (fused) checkpoint back to TorchScript, checking
# numerical parity against the trained branched model along the way.
set -euo pipefail

. ./path.sh
. ./cmd.sh

config=conf/adapter_only.yaml

. ./utils/parse_options.sh

mkdir -p exp/log

log_name=$(basename "${config}" .yaml)
${train_cmd} "exp/log/export_jit_${log_name}.log" \
  "${PYTHON}" -m silero.trainer.export_jit --config "${config}"
