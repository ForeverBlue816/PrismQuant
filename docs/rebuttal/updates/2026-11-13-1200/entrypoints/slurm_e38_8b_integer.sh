#!/bin/bash
#SBATCH --job-name=e38-8b-g128
#SBATCH --partition=cluster02
#SBATCH --account=rose
#SBATCH --qos=rose
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --no-requeue
#SBATCH --output=runs/e38-8b-g128-%j.out
#SBATCH --error=runs/e38-8b-g128-%j.err
# GPU, memory, walltime and any preceding Stage-A process are selected by root.
# This executable may be called AFTER Stage A exits within the same allocation;
# never run the two Python processes concurrently on one GPU.
set -euo pipefail
umask 0007
e38_code="${SLURM_SUBMIT_DIR:?Run from an allocated job submitted in the worktree}"
e38_data=/projects/nar/nar-validation
: "${E38_SOURCE_COMMIT:?record source revision}"
e38_rotation="${E38_ROTATION:-nar_k8}"
e38_seed="${E38_SEED:-0}"
e38_phase="${E38_PHASE:-equivalence-timing}"
e38_suffix="8b_${e38_rotation}_seed${e38_seed}_${e38_phase}_job${SLURM_JOB_ID}"
export HF_HOME="$e38_data/cache/huggingface" HF_DATASETS_CACHE="$e38_data/cache/datasets"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="/home/yanlongc/.e13_packages:$e38_code"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="$e38_data/tmp"
cd "$e38_code"
e38_extra=()
if [[ -n "${E38_CHECKPOINT_DIRECTORY:-}" ]]; then
  e38_extra+=(--checkpoint-directory "$E38_CHECKPOINT_DIRECTORY")
fi
e38_status=0
"$e38_data/venv/bin/python" -m nar.e38_8b_integer \
  --rotation "$e38_rotation" --seed "$e38_seed" --phase "$e38_phase" \
  --packed-cache "${E38_PACKED_CACHE:-$e38_data/artifacts/e38/packed_8b_${e38_rotation}_seed${e38_seed}}" \
  --artifact-dir "$e38_data/artifacts/e38/integer_${e38_suffix}" \
  --output "$e38_code/results/e38/raw/integer_${e38_suffix}.json" \
  --source-commit "$E38_SOURCE_COMMIT" --repetitions 20 --warmups 10 \
  --decode-steps "${E38_DECODE_STEPS:-128}" --discard-decode-steps 8 \
  --batch-size "${E38_TASK_BATCH_SIZE:-8}" "${e38_extra[@]}" || e38_status=$?
flock -w 60 "$e38_code/runs/collect_e38.lock" bash "$e38_code/scripts/collect_rebuttal_stage.sh" e38 || true
exit "$e38_status"
