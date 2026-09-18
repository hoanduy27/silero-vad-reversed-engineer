#!/usr/bin/env bash
# RepVGG-style adaptation (fine-tuning) of the Silero VAD encoder
# (silero.model_beautified.BranchedEncoderBlock) on human-reviewed webrtc-audio-audit
# labels. Two config variants under conf/ select adapter-only vs. full fine-tuning;
# see conf/adapter_only.yaml and conf/full_finetune.yaml.
set -euo pipefail

. ./path.sh
. ./cmd.sh

stage=1
stop_stage=100

dataset_dir=       # path to webrtc-audio-audit's labeling_dataset (contains manifest.json + audio/)
config=conf/adapter_only.yaml

. ./utils/parse_options.sh

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "stage 0: convert webrtc-audio-audit labeling_dataset (reviewed entries only)"
  local/prepare_data.sh --dataset-dir "${dataset_dir}"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "stage 1: train (config=${config})"
  local/train.sh --config "${config}"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "stage 2: FAR/FRR report (config=${config})"
  local/evaluate.sh --config "${config}"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "stage 3: export to TorchScript (config=${config})"
  local/export_jit.sh --config "${config}"
fi

echo "Done. See ${REPO_ROOT}/egs/vad/webrtc_audit/exp/"
