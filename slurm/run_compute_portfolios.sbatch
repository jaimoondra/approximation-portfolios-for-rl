#!/bin/bash
#SBATCH -J ndrl_portfolios              # Job name
#SBATCH -N 1                            # One node
#SBATCH --cpus-per-task=16              # CPUs for outer parallelism
#SBATCH --mem=32G                       # Memory
#SBATCH -t 08:00:00                     # Walltime hh:mm:ss
#SBATCH -o logs/ndrl_%j.out             # Stdout (relative to submission dir)
#SBATCH -e logs/ndrl_%j.err             # Stderr (relative to submission dir)
## If your cluster needs a partition or account, uncomment & set:
#SBATCH --partition=mit_normal          #sched_mit_sloan_batch  # Specify the Sloan batch partition
## #SBATCH -A <account>

# ========= USER CONFIG =========
CONDA_ENV="rl-portfolio"    # your conda env name
PROJECT_DIR="$HOME/approximation-portfolios-for-rl"
PY_SCRIPT="compute_portfolios.py"
# ==============================

set -euo pipefail

echo "[$(date)] Job $SLURM_JOB_ID starting on $(hostname)"
echo "Submit CWD: $(pwd)"
echo "Project: $PROJECT_DIR"
echo "Script : $PY_SCRIPT"

# ---- Load conda (adjust to your cluster) ----
module load anaconda 2>/dev/null || true
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
  source "/opt/conda/etc/profile.d/conda.sh"
fi
conda activate "$CONDA_ENV"

# ---- Runtime env ----
# Outer parallelism (Python spawns one process per score file)
export OUTER_WORKERS="${SLURM_CPUS_PER_TASK:-8}"
# Disable inner pools to avoid nested oversubscription
export N_WORKERS_INNER=1
# Prevent BLAS thread storms inside workers
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# Headless plotting
export MPLBACKEND=Agg

echo "CPUs requested: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "OUTER_WORKERS : $OUTER_WORKERS"
echo "N_WORKERS_INNER: $N_WORKERS_INNER"

# (Optional) print versions for reproducibility
python - <<'PY'
import sys, multiprocessing as mp
import numpy as np, pandas as pd, matplotlib
print("Python:", sys.version.split()[0])
print("NumPy :", np.__version__)
print("Pandas:", pd.__version__)
print("MPL   :", matplotlib.__version__)
print("CPUs  :", mp.cpu_count())
PY

# Move to notebook directory so relative paths in the script work
cd "$PROJECT_DIR/notebooks/natural_disaster"

# Create logs directory under the repo (after cd) if you want repo-local logs too
mkdir -p logs

# Run with srun to bind resources correctly
echo "[$(date)] Launching compute_portfolios.py ..."
srun python -u "$PY_SCRIPT"

echo "[$(date)] Done."
