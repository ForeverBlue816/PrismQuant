#!/bin/bash
#SBATCH --job-name=pq8b-fast-rotation
#SBATCH --partition=cluster02
#SBATCH --account=rose
#SBATCH --qos=rose
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=a40:1
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=runs/pq8b-fast-rotation-%j.out
#SBATCH --error=runs/pq8b-fast-rotation-%j.err
set -uo pipefail
root=/home/yanlongc/prismquant-rebuttal-e38
data=/projects/nar/nar-validation
cd "$root"
: "${E38_SOURCE_COMMIT:?}"
export PYTHONPATH="$root/quarot-llama3:$root"
export HF_HOME="$data/cache/huggingface" HF_DATASETS_CACHE="$data/cache/datasets"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export XDG_CACHE_HOME="$data/cache/xdg" TRITON_CACHE_DIR="$data/cache/triton-e28" TMPDIR="$data/tmp"
status=0
# Same trained PQ checkpoint and physical GPU for original/optimized controls.
# No FP16 row, no Hadamard row, no recalibration or altered quantization setting.
for backend in original fast; do
  "$data/e28-env/bin/python" -m e38.stage_a_8b --row nar_k8 --phase ppl-timing \
    --rotation-backend "$backend" --source-commit "$E38_SOURCE_COMMIT" \
    --artifact-dir "$data/artifacts/e38/native_packed8b_seed0_v1" \
    --repetitions 20 --warmup 10 --batch-size 1 \
    --output "$root/results/e38/raw/stage_a_8b_nar_k8_seed0_${backend}_job${SLURM_JOB_ID}.json" \
    >"$root/runs/pq8b_${backend}_job${SLURM_JOB_ID}.out" \
    2>"$root/runs/pq8b_${backend}_job${SLURM_JOB_ID}.err" || status=1
  flock -w 60 "$root/runs/collect_e38.lock" bash scripts/collect_rebuttal_stage.sh e38 || true
done
exit "$status"
