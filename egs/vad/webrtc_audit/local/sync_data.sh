#!/usr/bin/env bash
# Convenience wrapper for local/prepare_data.sh: this dataset is an ongoing,
# live project (the labeling app gets more recordings reviewed over time), so
# re-run this on demand to pull in whatever's newly reviewed. Only newly
# reviewed recordings are transcoded -- already-converted wavs are cached (see
# silero/data/convert_webrtc_audit.py's convert_audio) -- so re-syncing is cheap
# even as the source dataset grows.
set -euo pipefail

. ./path.sh
. ./cmd.sh

dataset_dir=/home/duy/github/olli/buddyos-omni-grpc/tools/webrtc-audio-audit/labeling_dataset

. ./utils/parse_options.sh

local/prepare_data.sh --dataset-dir "${dataset_dir}"
