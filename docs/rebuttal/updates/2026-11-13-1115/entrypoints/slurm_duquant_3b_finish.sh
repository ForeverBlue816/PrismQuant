#!/bin/bash
#SBATCH --job-name=duquant3b-winogrande
#SBATCH --partition=cluster02
#SBATCH --account=rose
#SBATCH --qos=rose
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=a40:1
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=runs/duquant3b-finish-%j.out
#SBATCH --error=runs/duquant3b-finish-%j.err
set -uo pipefail
root=/home/yanlongc/prismquant-rebuttal-e38
code=/home/yanlongc/prismquant-rebuttal-e42
export E42_CODE_DIR="$code" E42_MODEL=llama32_3b E42_DUQUANT_PHASE=eight-tasks E42_TASK_BATCH_SIZE=1
export E42_DUQUANT_ARTIFACT=/projects/nar/nar-validation/artifacts/e42/duquant_3b_seed0_v1
export E42_RESUME_TASKS_FROM="$code/results/e42/raw/duquant_3b_seed0_eight-tasks_retry2_job1736285.json"
export E42_OUTPUT_FILE="$code/results/e42/raw/duquant_3b_seed0_eight-tasks_finish_job${SLURM_JOB_ID}.json"
: "${E42_SOURCE_COMMIT:?}"
unset E42_RESUME_PREPARATION_FROM
bash "$code/slurm_e42_duquant_native.sh"
status=$?
flock -w 60 "$root/runs/collect_requested_baselines.lock" /projects/nar/nar-validation/venv/bin/python \
  "$root/scripts/collect_requested_baselines.py" || true
exit "$status"
