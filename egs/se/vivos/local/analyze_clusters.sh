#!/usr/bin/env bash
# Check whether visual t-SNE clusters correspond to distinct speaker groups.
set -euo pipefail

. ./path.sh
. ./cmd.sh

level=utterance
proj=
n_clusters=2

. ./utils/parse_options.sh

if [ -z "${proj}" ]; then
  if [ "${level}" == "frame" ]; then
    proj=exp/plots/tsne_speaker_embeddings_frame.proj.npz
  else
    proj=exp/plots/tsne_speaker_embeddings.proj.npz
  fi
fi

mkdir -p exp/log

${train_cmd} exp/log/analyze_clusters_${level}.log \
  "${PYTHON}" -m silero.speaker_sim.analyze_clusters \
    --level "${level}" \
    --proj "${proj}" \
    --n-clusters "${n_clusters}"
