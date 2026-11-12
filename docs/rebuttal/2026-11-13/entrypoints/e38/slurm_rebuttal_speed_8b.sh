#!/bin/bash
#SBATCH --job-name=pq8b-official-speed
#SBATCH --partition=cluster02
#SBATCH --account=rose
#SBATCH --qos=rose
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=a40:1
#SBATCH --mem=40G
#SBATCH --time=1-00:00:00
#SBATCH --no-requeue
#SBATCH --output=runs/rebuttal-speed8b-%j.out
#SBATCH --error=runs/rebuttal-speed8b-%j.err
set -uo pipefail
root=/home/yanlongc/prismquant-rebuttal-e38
data=/projects/nar/nar-validation
cd "$root"
: "${E38_SOURCE_COMMIT:?}" "${E39_SOURCE_COMMIT:?}" "${E42_SOURCE_COMMIT:?}"
status=0
# Sequential processes keep all methods on the same allocated physical A40.
# Shell children isolate Torch/CUDA/PYTHONPATH environments between methods.
# Every path evaluates full141 PPL on the same object later used for timing.
# All use B1, prefix2048, decode128/drop8, token100,10warm/20reps,last logits.
for cleanup in official remove-dead-random; do
  FLATQUANT_CACHE_OPTIMIZATION="$cleanup" bash /home/yanlongc/prismquant-rebuttal-e39/slurm_e39_native_8b.sh || status=1
done
(
  export PYTHONPATH="$root/quarot-llama3:$root"
  export HF_HOME="$data/cache/huggingface" HF_DATASETS_CACHE="$data/cache/datasets"
  export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
  export XDG_CACHE_HOME="$data/cache/xdg" TRITON_CACHE_DIR="$data/cache/triton-e28" TMPDIR="$data/tmp"
  "$data/e28-env/bin/python" -m e38.stage_a_8b --row nar_k8 --phase ppl-timing \
    --rotation-backend fast --cache-timing fresh --source-commit "$E38_SOURCE_COMMIT" \
    --artifact-dir "$data/artifacts/e38/native_packed8b_seed0_v1" \
    --repetitions 20 --warmup 10 --batch-size 1 \
    --output "$root/results/e38/raw/stage_a_8b_nar_k8_seed0_fast_officialcomparison_job${SLURM_JOB_ID}.json"
) >"$root/runs/pq8b_fast_officialcomparison_job${SLURM_JOB_ID}.out" \
  2>"$root/runs/pq8b_fast_officialcomparison_job${SLURM_JOB_ID}.err" || status=1
for method in flatquant duquant; do
  bash scripts/rebuttal_baseline_model_worker.sh llama31_8b "$method" timing || status=1
done
flock -w 60 "$root/runs/collect_e38.lock" bash scripts/collect_rebuttal_stage.sh e38 || true
"$data/venv/bin/python" scripts/collect_native_speed_8b.py --job-id "$SLURM_JOB_ID" || status=1
exit "$status"
