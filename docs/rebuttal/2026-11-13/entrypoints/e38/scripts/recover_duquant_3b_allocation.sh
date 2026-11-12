#!/bin/bash
# Run as a new single-GPU step in the existing, user-owned 3B allocation.
set -uo pipefail
root=/home/yanlongc/prismquant-rebuttal-e38
code=/home/yanlongc/prismquant-rebuttal-e42
data=/projects/nar/nar-validation
: "${SLURM_JOB_ID:?existing allocation required}" "${E42_SOURCE_COMMIT:?}"
export E42_CODE_DIR="$code" E42_MODEL=llama32_3b
export E42_DUQUANT_ARTIFACT="$data/artifacts/e42/duquant_3b_seed0_v1"
export E42_TASK_BATCH_SIZE=1
status=0
for phase in quantize-ppl eight-tasks; do
  export E42_DUQUANT_PHASE="$phase"
  export E42_OUTPUT_FILE="$code/results/e42/raw/duquant_3b_seed0_${phase}_retry1_job${SLURM_JOB_ID}.json"
  if [[ "$phase" == quantize-ppl ]]; then
    export E42_RESUME_PREPARATION_FROM="$code/results/e42/raw/duquant_3b_seed0_quantize-ppl_job1736285.json"
  else
    unset E42_RESUME_PREPARATION_FROM
  fi
  bash "$code/slurm_e42_duquant_native.sh" \
    >"$root/runs/baseline_3b_duquant_${phase}_retry1_job${SLURM_JOB_ID}.out" \
    2>"$root/runs/baseline_3b_duquant_${phase}_retry1_job${SLURM_JOB_ID}.err" || status=1
  flock -w 60 "$root/runs/collect_requested_baselines.lock" "$data/venv/bin/python" \
    "$root/scripts/collect_requested_baselines.py" || true
done
exit "$status"
