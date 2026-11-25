#!/bin/bash
# One-GPU supervised fine-tuning job for jevtw.sft.
#
#   sbatch --job-name=E0010b-s17 scripts/slurm/sft.sbatch <config.json> <out_dir> [extra jevtw.sft args...]
# Submit from the repository root (mkdir -p runs/slurm first). Offline mode as in train.sbatch.
#SBATCH --account=aip-rjliao
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=runs/slurm/%x-%j.out
#SBATCH --error=runs/slurm/%x-%j.out
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: sbatch [--job-name=NAME] scripts/slurm/sft.sbatch <config.json> <out_dir> [sft args...]" >&2
    exit 2
fi
CONFIG=$1
OUT=$2
shift 2

REPO=${SLURM_SUBMIT_DIR:-$PWD}
cd "$REPO"
if [ -e "$OUT" ]; then            # CLAUDE.md: never overwrite an existing run directory
    echo "output directory $OUT already exists; choose a new one" >&2
    exit 2
fi

set +u                            # the module scripts read unset variables
source /etc/profile
module load python/3.12 arrow cuda
source "$REPO/.venv/bin/activate"
set -u

export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "job $SLURM_JOB_ID ($SLURM_JOB_NAME) on $(hostname), started $(date -Is)"
echo "commit $(git rev-parse HEAD) on $(git rev-parse --abbrev-ref HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
# jevtw.train falls back to the CPU when CUDA is missing; a GPU job must stop instead.
python -c "import sys, torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 1)"

START=$(date +%s)
python -m jevtw.sft --config "$CONFIG" --out "$OUT" "$@"
echo "finished $(date -Is), wall time $(( $(date +%s) - START )) s"
