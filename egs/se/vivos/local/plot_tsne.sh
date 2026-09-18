#!/usr/bin/env bash
# t-SNE projection of the speaker embeddings, colored by speaker.
set -euo pipefail

. ./path.sh
. ./cmd.sh

level=utterance
embeddings=
out=

. ./utils/parse_options.sh

if [ -z "${embeddings}" ]; then
  if [ "${level}" == "frame" ]; then
    embeddings=exp/embeddings/embeddings_frame.npz
  else
    embeddings=exp/embeddings/embeddings.npz
  fi
fi

if [ -z "${out}" ]; then
  if [ "${level}" == "frame" ]; then
    out=exp/plots/tsne_speaker_embeddings_frame.png
  else
    out=exp/plots/tsne_speaker_embeddings.png
  fi
fi

mkdir -p exp/log "$(dirname "${out}")"

${train_cmd} exp/log/plot_tsne_${level}.log \
  "${PYTHON}" -m silero.speaker_sim.plot_tsne \
    --level "${level}" \
    --embeddings "${embeddings}" \
    --out "${out}"
