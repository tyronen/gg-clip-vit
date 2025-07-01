#!/usr/bin/env bash
# run like `source setup.sh` on any remote to ensure active shell is set up with venv

# --- 1. System Setup ---
apt-get install -y vim rsync git nvtop htop tmux curl ca-certificates git-lfs lsof nano
cp ~/.env .env
set -a; source .env; set +a
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
if [[ -n "$SSH_CONNECTION" && -d /workspace ]]; then
  echo "🐧 Running on remote runpod with storage attached - setting custom uv/hf cache dir"
  mkdir -p /workspace/.cache/uv /workspace/.cache/datasets_cache
  set -a
  export UV_CACHE_DIR="/workspace/.cache/uv"
  export HF_DATASETS_CACHE="/workspace/.cache/datasets_cache"
  set +a
fi

# --- 2. Create Virtual Environment ---
echo "🐍 Creating a clean virtual environment..."
uv venv --python /usr/bin/python3.10 .venv

# --- 3. DYNAMIC DEPENDENCY DETECTION ---
# Detects the pre-installed torch/numpy versions to ensure driver compatibility.
echo "🔍 Detecting system-installed torch and numpy versions..."
PY_SYSTEM="/usr/bin/python3.10"
TORCH_VER=$($PY_SYSTEM -c "import torch; print(torch.__version__)")
NUMPY_VER=$($PY_SYSTEM -c "import numpy; print(numpy.__version__)")

if [[ -z "$TORCH_VER" ]]; then
    echo "❌ Critical Error: Could not detect system-installed torch. Aborting."
    exit 1
fi
echo "  ✅ Detected System PyTorch: $TORCH_VER"
echo "  ✅ Detected System NumPy:   $NUMPY_VER"

# --- 4. INJECT DYNAMIC DEPENDENCIES ---
# Installs ONLY the detected versions into the new venv. This locks them in.
echo "💉 Injecting detected torch and numpy versions into .venv..."
uv pip install "torch==$TORCH_VER" "numpy==$NUMPY_VER"

# --- 5. Install Static Dependencies ---
# Syncs the default dependencies from pyproject.toml.
# It will NOT install the 'local' group.
# It will see numpy is already installed and will not overwrite it.
echo "📦 Syncing remaining dependencies from pyproject.toml..."
uv sync

# --- 6. Activate the Environment ---
echo "✅ Environment setup complete! Activating .venv..."
source .venv/bin/activate