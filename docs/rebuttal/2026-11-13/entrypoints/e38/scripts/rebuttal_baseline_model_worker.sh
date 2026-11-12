#!/bin/bash
set -uo pipefail
root=/home/yanlongc/prismquant-rebuttal-e38
data=/projects/nar/nar-validation
model="${1:?model required}"
method="${2:?method required}"
mode="${3:-accuracy}"
case "$model" in llama31_8b) short=8b;; llama32_3b) short=3b;; *) exit 2;; esac
case "$mode" in accuracy) ;; timing) ;; *) exit 2;; esac
export HF_HOME="$data/cache/huggingface" HF_DATASETS_CACHE="$data/cache/datasets"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TMPDIR="$data/tmp" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$root/runs"
status=0
run_phase() {
  local phase="$1";shift
  "$@" >"$root/runs/baseline_${short}_${method}_${phase}_job${SLURM_JOB_ID}.out" \
    2>"$root/runs/baseline_${short}_${method}_${phase}_job${SLURM_JOB_ID}.err"
  local rc=$?
  python3 - "$root" "$model" "$method" "$phase" "$rc" "$SLURM_JOB_ID" <<'PY'
import datetime,json,sys
from pathlib import Path
p=Path(sys.argv[1])/'runs'/('baseline_stages_'+sys.argv[6]+'.jsonl')
with p.open('a') as f:f.write(json.dumps(dict(model=sys.argv[2],method=sys.argv[3],phase=sys.argv[4],exit_code=int(sys.argv[5]),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))+'\n')
PY
  (( rc == 0 )) || status=1
  flock -w 60 "$root/runs/collect_requested_baselines.lock" "$data/venv/bin/python" \
    "$root/scripts/collect_requested_baselines.py" >>"$root/runs/collect_baselines_job${SLURM_JOB_ID}.log" 2>&1 || true
}
if [[ "$method" == flatquant ]]; then
  code=/home/yanlongc/prismquant-rebuttal-e39
  cd "$code"
  export PYTHONPATH="/home/yanlongc/.prismquant/llama31-baseline-packages:/home/yanlongc/FlatQuant-upstream:/home/yanlongc/.e13_packages:$code"
  phases=(train-ppl tasks)
  [[ "$mode" != timing ]] || phases=(ppl-timing)
  for phase in "${phases[@]}"; do
    run_phase "$phase" "$data/venv/bin/python" -m nar.e39_flatquant --model "$model" \
      --checkpoint "$data/checkpoints/rebuttal_${short}/flatquant_native_seed0" --phase "$phase" \
      --source-commit "$E39_SOURCE_COMMIT" \
      --output "$code/results/e39/raw/native_${short}_seed0_${phase}_job${SLURM_JOB_ID}.json"
  done
elif [[ "$method" == duquant ]]; then
  code=/home/yanlongc/prismquant-rebuttal-e42
  cd "$code"
  export PYTHONPATH="/home/yanlongc/.prismquant/llama31-baseline-packages:/home/yanlongc/DuQuant-upstream:/home/yanlongc/.e13_packages:$code"
  phases=(quantize-ppl eight-tasks)
  [[ "$mode" != timing ]] || phases=(ppl-timing)
  for phase in "${phases[@]}"; do
    run_phase "$phase" "$data/venv/bin/python" -m nar.e42_duquant_native --model "$model" \
      --artifact-dir "$data/artifacts/e42/duquant_${short}_seed0_v1" --phase "$phase" \
      --batch-size 1 --source-commit "$E42_SOURCE_COMMIT" \
      --output "$code/results/e42/raw/duquant_${short}_seed0_${phase}_job${SLURM_JOB_ID}.json"
  done
else
  exit 2
fi
exit "$status"
