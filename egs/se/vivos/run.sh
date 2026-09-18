#!/usr/bin/env bash
# Kaldi-style recipe: can the SileroVAD encoder (silero/model_beautified.py)
# double as a speaker encoder? Embed VIVOS utterances with the encoder only
# (VADRNN.forward_embed, decoder RNN bypassed) and visualize with t-SNE,
# colored by speaker.
#
# --level utterance (default): one mean-pooled embedding per utterance.
# --level frame: every encoder time step kept as its own point
#   (forward_embed(..., return_individuals=True)).
set -euo pipefail

. ./path.sh
. ./cmd.sh

stage=0
stop_stage=100

split=train
num_speakers=10
utts_per_speaker=5
level=utterance

. ./utils/parse_options.sh

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "stage 0: download VIVOS and prepare a subset manifest"
  local/download_and_prepare.sh \
    --split "${split}" \
    --num-speakers "${num_speakers}" \
    --utts-per-speaker "${utts_per_speaker}" \
    --out-dir exp/data
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "stage 1: compute encoder-only speaker embeddings (level=${level})"
  local/compute_embeddings.sh --level "${level}" --manifest exp/data/manifest.csv
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "stage 2: t-SNE plot of speaker embeddings (level=${level})"
  local/plot_tsne.sh --level "${level}"
fi

echo "Done. See ${REPO_ROOT}/egs/se/vivos/exp/plots/"
