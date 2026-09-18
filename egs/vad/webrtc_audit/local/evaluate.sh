#!/usr/bin/env bash
# FAR/FRR report on the held-out val split: fine-tuned (fused) checkpoint vs.
# the original, un-adapted one.
set -euo pipefail

. ./path.sh
. ./cmd.sh

config=conf/adapter_only.yaml
checkpoint=
threshold=0.5

. ./utils/parse_options.sh

mkdir -p exp/log

checkpoint_arg=()
if [ -n "${checkpoint}" ]; then
  checkpoint_arg=(--checkpoint "${checkpoint}")
fi

log_name=$(basename "${config}" .yaml)
${train_cmd} "exp/log/evaluate_${log_name}.log" \
  "${PYTHON}" -m silero.trainer.evaluate --config "${config}" --threshold "${threshold}" "${checkpoint_arg[@]}"
