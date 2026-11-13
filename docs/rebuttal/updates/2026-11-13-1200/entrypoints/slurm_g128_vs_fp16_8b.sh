#!/bin/bash
#SBATCH --job-name=pq8b-g128-vs-fp16
#SBATCH --partition=cluster02
#SBATCH --account=rose
#SBATCH --qos=rose
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=a40:1
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=runs/g128-vs-fp16-8b-%j.out
#SBATCH --error=runs/g128-vs-fp16-8b-%j.err
set -uo pipefail
umask 0007
root=/home/yanlongc/prismquant-rebuttal-e38
data=/projects/nar/nar-validation
cd "$root"
: "${E38_SOURCE_COMMIT:?record source revision}"
status=0
# Sequential processes share one allocated A40. Each checks physical GPU
# exclusivity throughout timing. Both evaluate full141 PPL on the same loaded
# model object used for timing, with fresh cache construction inside prefill.
(
  export PYTHONPATH="$root/quarot-llama3:$root"
  export HF_HOME="$data/cache/huggingface" HF_DATASETS_CACHE="$data/cache/datasets"
  export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
  export XDG_CACHE_HOME="$data/cache/xdg" TRITON_CACHE_DIR="$data/cache/triton-e28" TMPDIR="$data/tmp"
  "$data/e28-env/bin/python" -m e38.stage_a_8b --row fp16 --phase ppl-timing \
    --cache-timing fresh --source-commit "$E38_SOURCE_COMMIT" \
    --artifact-dir "$data/artifacts/e38/native_packed8b_seed0_v1" \
    --repetitions 20 --warmup 10 --batch-size 1 \
    --output "$root/results/e38/raw/stage_a_8b_fp16_seed0_g128comparison_job${SLURM_JOB_ID}.json"
) >"$root/runs/fp16_g128comparison_job${SLURM_JOB_ID}.out" \
  2>"$root/runs/fp16_g128comparison_job${SLURM_JOB_ID}.err" || status=1
# Existing g128 prototype: packed W4 storage expanded to INT8 group GEMMs;
# BF16 cache plus KV4 simulation. This is not a native INT4 speedup claim.
E38_PHASE=ppl-timing E38_ROTATION=nar_k8 E38_SEED=0 E38_TASK_BATCH_SIZE=1 \
  bash "$root/slurm_e38_8b_integer.sh" \
  >"$root/runs/g128_ppl_timing_job${SLURM_JOB_ID}.out" \
  2>"$root/runs/g128_ppl_timing_job${SLURM_JOB_ID}.err" || status=1
python3 "$root/scripts/collect_g128_vs_fp16.py" --job-id "$SLURM_JOB_ID" || status=1
exit "$status"
