#!/bin/bash
#SBATCH --job-name=dp_shift
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%u/%x_%j.out
#SBATCH --error=logs/%u/%x_%j.err

set -euo pipefail

# Submit from the repository root.
# mkdir -p "logs/$(id -un)" && sbatch engaging/submit_experiment.slurm
# Optional arguments are the config path and an explicit results directory.
cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch from the repository root}"
umask 077

CONFIG="${1:-configs/experiment.json}"
OUTPUT="${2:-results/$(id -un)/run_${SLURM_JOB_ID:?Missing Slurm job ID}}"
WORKERS="${SLURM_CPUS_PER_TASK:?Request CPUs with --cpus-per-task}"
PYTHON="${DP_SHIFT_VENV:-.venv}/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "Python environment not found at $PYTHON. Set DP_SHIFT_VENV to a cluster virtual environment." >&2
    exit 1
fi
if [[ ! -r "$CONFIG" ]]; then
    echo "Cannot read config: $CONFIG" >&2
    exit 1
fi

# Each comparison worker uses one numerical thread.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date) with $WORKERS CPU workers"
echo "Config: $CONFIG"
echo "Output: $OUTPUT"

srun --ntasks=1 --cpus-per-task="$WORKERS" "$PYTHON" -u -m dp_shift experiment \
    --config "$CONFIG" --output "$OUTPUT" --workers "$WORKERS"

echo "Experiment finished at $(date)"
