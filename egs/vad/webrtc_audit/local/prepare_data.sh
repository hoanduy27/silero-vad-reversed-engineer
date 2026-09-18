#!/usr/bin/env bash
# Convert the webrtc-audio-audit labeling_dataset (reviewed entries only) into
# this project's wav + manifest.jsonl format. See silero/data/convert_webrtc_audit.py.
set -euo pipefail

. ./path.sh
. ./cmd.sh

dataset_dir=
out_dir=exp/data/webrtc_audit

. ./utils/parse_options.sh

if [ -z "${dataset_dir}" ]; then
  echo "$0: --dataset-dir is required (path to webrtc-audio-audit's labeling_dataset)" >&2
  exit 1
fi

mkdir -p exp/log "$(dirname "${out_dir}")"

${train_cmd} exp/log/prepare_data.log \
  "${PYTHON}" -m silero.data.convert_webrtc_audit \
    --dataset-dir "${dataset_dir}" \
    --out-dir "${out_dir}"
