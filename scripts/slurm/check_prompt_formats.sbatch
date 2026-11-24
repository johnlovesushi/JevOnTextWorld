#!/bin/bash
# One-GPU run of scripts/check_prompt_formats.py (submit from the repository root; mkdir -p runs/slurm first).
#
#   sbatch scripts/slurm/check_prompt_formats.sbatch [check_prompt_formats.py args...]   (runs it with --sample)
#SBATCH --job-name=T0006-check
#SBATCH --account=aip-rjliao
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=runs/slurm/%x-%j.out
#SBATCH --error=runs/slurm/%x-%j.out
set -euo pipefail
REPO=${SLURM_SUBMIT_DIR:-$PWD}
cd "$REPO"
set +u
source /etc/profile
module load python/3.12 arrow cuda
source "$REPO/.venv/bin/activate"
set -u
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
echo "job $SLURM_JOB_ID on $(hostname), commit $(git rev-parse HEAD) ($(git status --porcelain | wc -l) uncommitted files)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python scripts/check_prompt_formats.py --sample "$@"
