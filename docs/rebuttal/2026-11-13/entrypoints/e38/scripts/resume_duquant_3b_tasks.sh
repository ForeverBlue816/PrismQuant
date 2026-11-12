#!/bin/bash
# Reuse the completed quantized checkpoint after an interrupted task-eval step.
set -uo pipefail
root=/home/yanlongc/prismquant-rebuttal-e38
code=/home/yanlongc/prismquant-rebuttal-e42
export E42_CODE_DIR="$code" E42_MODEL=llama32_3b E42_DUQUANT_PHASE=eight-tasks E42_TASK_BATCH_SIZE=1
export E42_DUQUANT_ARTIFACT=/projects/nar/nar-validation/artifacts/e42/duquant_3b_seed0_v1
export E42_OUTPUT_FILE="$code/results/e42/raw/duquant_3b_seed0_eight-tasks_retry2_job${SLURM_JOB_ID}.json"
unset E42_RESUME_PREPARATION_FROM
bash "$code/slurm_e42_duquant_native.sh" \
  >"$root/runs/baseline_3b_duquant_eight-tasks_retry2_job${SLURM_JOB_ID}.out" \
  2>"$root/runs/baseline_3b_duquant_eight-tasks_retry2_job${SLURM_JOB_ID}.err"
status=$?
flock -w 60 "$root/runs/collect_requested_baselines.lock" /projects/nar/nar-validation/venv/bin/python \
  "$root/scripts/collect_requested_baselines.py" || true
exit "$status"
