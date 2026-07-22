#!/bin/bash
# =============================================================================
# SLURM job script — zero-touch-fl adversarial GRPO training on the GPU cluster.
#
# Submit from the repo root on a login node (deie01 / deie02):
#     mkdir -p logs/slurm && sbatch train.sh
#
# Track it:            squeue -u $USER
# Watch the log:       tail -f logs/slurm/<job-name>_<jobid>.out
# Cancel it:           scancel <jobid>
# After it finishes:   sacct -j <jobid>
#
# The training RESUMES automatically on resubmit: existing adapters +
# checkpoints/rl_progress.json are reloaded (see README "Resume"). So if the
# --time cap ends the job mid-run, just `sbatch train.sh` again to continue.
# =============================================================================

#SBATCH --job-name=49_Dinuth_ztfl          # CHANGE to GroupNo_StudentName_AnyOther
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1                       # one GPU (type-agnostic; deie01 has 1 GPU)
#SBATCH --cpus-per-task=4                  # MNIST data loading / preprocessing
#SBATCH --mem=16G
#SBATCH --time=24:00:00                    # safety cap (partition limit is infinite; raise/remove as needed)
#SBATCH --output=logs/slurm/%x_%j.out      # %x = job-name, %j = job id
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

# --- Run from the directory the job was submitted from --------------------
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/slurm

# --- Activate a Python environment ----------------------------------------
# Prefer the project's own ./venv (built by setup_env.sh). If it doesn't exist,
# fall back to whatever env is already active on the node (e.g. the shared
# ai_env). Override the path with:  ZTFL_ENV=/path/to/venv sbatch train.sh
ENV_DIR="${ZTFL_ENV:-$SLURM_SUBMIT_DIR/venv}"
if [ -f "$ENV_DIR/bin/activate" ]; then
    echo "[env] activating project venv: $ENV_DIR"
    # shellcheck disable=SC1091
    source "$ENV_DIR/bin/activate"
else
    echo "[env] no ./venv found — using the pre-activated environment: ${VIRTUAL_ENV:-${CONDA_DEFAULT_ENV:-system}}"
fi

# Fail early with a clear message if the RL stack isn't importable.
python -c "import torch, unsloth" 2>/dev/null || {
    echo "ERROR: torch/unsloth not importable in this environment." >&2
    echo "       Build the project venv once:  bash setup_env.sh  (see its header)." >&2
    exit 1
}

# --- Provenance / sanity in the log ---------------------------------------
echo "=================================================================="
echo "Job          : $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "Node         : $(hostname)"
echo "GPUs (SLURM) : ${CUDA_VISIBLE_DEVICES:-none}"
echo "Started      : $(date)"
echo "=================================================================="
nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

# --- Train ----------------------------------------------------------------
# Full adversarial GRPO training (Phase 1 runs once, then the RL arms race).
# To bound a smoke run instead, set ROUNDS, e.g.:  ROUNDS=8 sbatch train.sh
if [ -n "${ROUNDS:-}" ]; then
    echo "[run] python main.py --env linux --rounds $ROUNDS"
    srun python main.py --env linux --rounds "$ROUNDS"
else
    echo "[run] python main.py --env linux"
    srun python main.py --env linux
fi

echo "Finished     : $(date)"
