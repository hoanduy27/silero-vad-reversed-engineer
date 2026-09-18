export REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)
export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
# uv-managed project venv (see pyproject.toml/uv.lock at REPO_ROOT); falls
# back to system python3 if it hasn't been synced yet (`uv sync` at REPO_ROOT).
if [ -x "${REPO_ROOT}/.venv/bin/python3" ]; then
  export PYTHON="${REPO_ROOT}/.venv/bin/python3"
else
  export PYTHON=python3
fi
