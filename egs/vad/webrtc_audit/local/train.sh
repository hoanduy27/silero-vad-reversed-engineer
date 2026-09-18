#!/usr/bin/env bash
# Fine-tune the VAD encoder per a config (conf/adapter_only.yaml or conf/full_finetune.yaml).
set -euo pipefail

. ./path.sh
. ./cmd.sh

config=conf/adapter_only.yaml
exp_dir=

. ./utils/parse_options.sh

mkdir -p exp/log

exp_dir_arg=()
if [ -n "${exp_dir}" ]; then
  exp_dir_arg=(--exp-dir "${exp_dir}")
fi

log_name=$(basename "${config}" .yaml)
${train_cmd} "exp/log/train_${log_name}.log" \
  "${PYTHON}" -m silero.trainer.trainer --config "${config}" "${exp_dir_arg[@]}"
