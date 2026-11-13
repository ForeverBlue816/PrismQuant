#!/bin/bash
#SBATCH --job-name=e39-flat-native8b
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=runs/e39-native8b-%j.out
#SBATCH --error=runs/e39-native8b-%j.err
set -euo pipefail
umask 0007
code_dir="${NAR_CODE_DIR:-/home/yanlongc/prismquant-rebuttal-e39}"
source /home/yanlongc/.prismquant/flatquant-native-tools/environment_torch26.sh
export PYTHONPATH="$PYTHONPATH:/home/yanlongc/.e13_packages:$code_dir"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$code_dir"
checkpoint="${FLATQUANT_SOURCE_CHECKPOINT:-/projects/nar/nar-validation/checkpoints/rebuttal_8b/flatquant_native_seed0}"
packed="${FLATQUANT_PACKED_CHECKPOINT:-/projects/nar/nar-validation/checkpoints/rebuttal_8b/flatquant_packed_native_seed0_v1}"
cache_optimization="${FLATQUANT_CACHE_OPTIMIZATION:-official}"
output="${FLATQUANT_NATIVE_OUTPUT:-$code_dir/results/e39/raw/packed_native_8b_seed0_${cache_optimization}_ppl-timing_job${SLURM_JOB_ID}.json}"
# No secondary process is launched on this allocation. The timing helper also
# checks the physical GPU process list before every recorded repetition.
/projects/nar/nar-validation/e28-env/bin/python -m nar.e39_flatquant_native \
  --checkpoint "$checkpoint" --packed-checkpoint "$packed" --output "$output" \
  --cache-optimization "$cache_optimization" \
  --source-commit "$(git rev-parse HEAD)"
