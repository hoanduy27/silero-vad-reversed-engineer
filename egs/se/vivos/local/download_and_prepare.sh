#!/usr/bin/env bash
# Download VIVOS and build a small (speakers x utterances) subset manifest.
set -euo pipefail

. ./path.sh
. ./cmd.sh

split=train
num_speakers=10
utts_per_speaker=5
out_dir=exp/data

. ./utils/parse_options.sh

mkdir -p exp/log

${train_cmd} exp/log/download_and_prepare.log \
  "${PYTHON}" -m silero.speaker_sim.download_vivos \
    --split "${split}" \
    --num-speakers "${num_speakers}" \
    --utts-per-speaker "${utts_per_speaker}" \
    --out-dir "${out_dir}"
