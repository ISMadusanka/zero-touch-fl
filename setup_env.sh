#!/bin/bash
# =============================================================================
# One-time environment setup for zero-touch-fl on the GPU cluster (deie01).
#
# Run this ONCE, on deie01, from inside the copied project directory. Because
# deie01 is both the login node and the single GPU node, a GPU is available
# right here — but to be a good cluster citizen, do the build inside an
# interactive allocation:
#
#     ssh student@10.50.20.197                      # deie01
#     cd ~/zero-touch-fl                            # the copied project
#     srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=16G --pty bash
#     bash setup_env.sh
#     exit                                          # release the allocation
#
# It creates ./venv INSIDE the project — exactly the path train.sh activates.
# After this, train.sh just sources ./venv — no reinstalling per job.
# =============================================================================
set -euo pipefail

# Create the venv inside the project by default (matches train.sh).
ENV_DIR="${ZTFL_ENV:-$PWD/venv}"

echo "[setup] Creating virtual environment at: $ENV_DIR"
python3 -m venv "$ENV_DIR"
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

echo "[setup] Upgrading pip"
pip install --upgrade pip

echo "[setup] Installing project requirements (this pulls the CUDA RL stack)"
pip install -r requirements.txt

echo "[setup] Sanity check: torch + CUDA visibility"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

echo "[setup] Done. In job scripts, activate with:  source $ENV_DIR/bin/activate"
