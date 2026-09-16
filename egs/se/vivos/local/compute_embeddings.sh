#!/usr/bin/env bash
# Encoder-only forward pass (VADRNN.forward_embed) over every utterance in
# the manifest to produce speaker embeddings, at either "utterance"
# (mean-pooled) or "frame" (return_individuals=True) granularity.
set -euo pipefail

. ./path.sh
. ./cmd.sh

level=utterance
manifest=exp/data/manifest.csv
out=

. ./utils/parse_options.sh

if [ -z "${out}" ]; then
  if [ "${level}" == "frame" ]; then
    out=exp/embeddings/embeddings_frame.npz
  else
    out=exp/embeddings/embeddings.npz
  fi
fi

mkdir -p exp/log "$(dirname "${out}")"

${train_cmd} exp/log/compute_embeddings_${level}.log \
  "${PYTHON}" -m silero.speaker_sim.compute_embeddings \
    --level "${level}" \
    --manifest "${manifest}" \
    --out "${out}"
