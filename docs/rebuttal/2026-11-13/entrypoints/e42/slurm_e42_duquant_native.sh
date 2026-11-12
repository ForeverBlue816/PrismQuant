#!/bin/bash
#SBATCH --job-name=e42-duquant-8b
#SBATCH --partition=cluster02
#SBATCH --account=rose
#SBATCH --qos=rose
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --no-requeue
#SBATCH --output=runs/e42-duquant-%j.out
#SBATCH --error=runs/e42-duquant-%j.err
# Root selects GPU/RAM/time or invokes this fragment with one visible GPU from
# the combined multi-GPU job. Never expose multiple GPUs to official DuQuant.
set -euo pipefail
umask 0007
e42_code="${E42_CODE_DIR:-${SLURM_SUBMIT_DIR:?submit from E42 worktree}}"
e42_data=/projects/nar/nar-validation
e42_phase="${E42_DUQUANT_PHASE:-quantize-ppl}"
e42_model="${E42_MODEL:-llama31_8b}"
case "$e42_model" in
  llama31_8b) e42_size=8b ;;
  llama32_3b) e42_size=3b ;;
  *) echo "Unsupported E42 model: $e42_model" >&2; exit 2 ;;
esac
: "${E42_SOURCE_COMMIT:?record frozen source revision}"
export PYTHONPATH="/home/yanlongc/.prismquant/llama31-baseline-packages:/home/yanlongc/DuQuant-upstream:/home/yanlongc/.e13_packages:$e42_code"
export HF_HOME="$e42_data/cache/huggingface" HF_DATASETS_CACHE="$e42_data/cache/datasets"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TMPDIR="$e42_data/tmp"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$e42_code"
e42_resume_args=()
if [[ -n "${E42_RESUME_PREPARATION_FROM:-}" ]]; then
  e42_resume_args=(--resume-preparation-from "$E42_RESUME_PREPARATION_FROM")
fi
exec "$e42_data/venv/bin/python" -m nar.e42_duquant_native \
 --phase "$e42_phase" --model "$e42_model" --source-commit "$E42_SOURCE_COMMIT" \
 --artifact-dir "${E42_DUQUANT_ARTIFACT:-$e42_data/artifacts/e42/duquant_${e42_size}_seed0_v1}" \
 --batch-size "${E42_TASK_BATCH_SIZE:-1}" \
 "${e42_resume_args[@]}" \
 --output "${E42_OUTPUT_FILE:-$e42_code/results/e42/raw/duquant_${e42_size}_seed0_${e42_phase}_job${SLURM_JOB_ID}.json}"
