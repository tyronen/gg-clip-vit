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
uv venv --system-site-packages

# --- 5. Install Dependencies from the Lock File ---
# This installs the exact same package versions every time, ensuring reproducibility.
echo "📦 Installing dependencies from lock file..."
uv sync --no-install-package torch --no-install-package numpy

# --- 6. Activate the Environment ---
echo "✅ Environment setup complete! Activating .venv..."
source .venv/bin/activate