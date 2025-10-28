#!/usr/bin/env bash
set -euo pipefail

# Configure uv-managed virtualenv to use ROCm PyTorch wheels.
# Usage: run from the project root after cloning the repository.

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but not found on PATH." >&2
  exit 1
fi

if [ ! -f "pyproject.toml" ]; then
  echo "Run this script from the project root containing pyproject.toml." >&2
  exit 1
fi

echo "[1/4] Initializing uv environment with CPython 3.11..."
uv run --python 3.11 python -c 'import sys; print(sys.version)'

echo "[2/4] Removing CUDA-flavored torch packages (if present)..."
uv pip uninstall --python .venv/bin/python torch torchvision torchaudio || true

echo "[3/4] Installing ROCm torch wheel..."
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/rocm6.1 torch==2.6.0

echo "[4/4] Verifying ROCm torch availability..."
UV_NO_SYNC=1 uv run --python 3.11 python -c 'import torch; print(torch.__version__, torch.version.hip if hasattr(torch.version, "hip") else None, torch.cuda.device_count(), torch.cuda.is_available())'

cat <<'EOM'

Success. Use "UV_NO_SYNC=1 uv run --python 3.11 ..." to keep uv from reinstalling CUDA wheels.
EOM
