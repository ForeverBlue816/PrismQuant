#!/bin/bash
#SBATCH --job-name=nar-e36-stats
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --qos=rose
#SBATCH --output=runs/e36-stats-%j.out
#SBATCH --error=runs/e36-stats-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export HF_HOME=/projects/nar/nar-validation/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
/projects/nar/nar-validation/venv/bin/python nar/e36_analysis.py --model "${1:?model}"
