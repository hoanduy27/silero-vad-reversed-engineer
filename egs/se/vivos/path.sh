export REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)
export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
export PYTHON=python3
