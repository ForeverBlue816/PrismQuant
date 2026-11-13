#!/bin/bash
set -uo pipefail
umask 0007
worker="${1:?worker number required}"
# User explicitly excludes OffQ without public official code. Preserve history, never execute worker2.
if [[ "$worker" == 2 ]]; then
  echo "OffQ/E40 disabled by user: no public official code; no experiment launched." >&2
  exit 64
fi
data=/projects/nar/nar-validation
root=/home/yanlongc/prismquant-rebuttal-e38
export HF_HOME="$data/cache/huggingface" HF_DATASETS_CACHE="$data/cache/datasets"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TMPDIR="$data/tmp" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
status=0
run_stage() {
  local label="$1"; shift
  "$@" >"$root/runs/compute_${label}_job${SLURM_JOB_ID}.out" 2>"$root/runs/compute_${label}_job${SLURM_JOB_ID}.err"
  local rc=$?
  python3 - "$root" "$worker" "$label" "$rc" "$SLURM_JOB_ID" <<'PY'
import json,sys,datetime
from pathlib import Path
p=Path(sys.argv[1])/'runs'/('compute_worker'+sys.argv[2]+'_'+sys.argv[5]+'.jsonl')
with p.open('a') as f:f.write(json.dumps(dict(stage=sys.argv[3],exit_code=int(sys.argv[4]),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))+'\n')
PY
  if (( rc != 0 )); then status=1; fi
  if (( rc == 0 )) && [[ "$label" == e39_train-ppl || "$label" == e42_quantize-ppl ]]; then
    # Release the existing comparison once both real 8B artifacts are available;
    # eight-task evaluation continues independently on the training allocation.
    python3 "$root/scripts/release_native_speed_when_ready.py" --release --trigger "$label" \
      >>"$root/runs/native_speed_release_job${SLURM_JOB_ID}.log" 2>&1 || true
  fi
  local experiment="${label%%_*}"
  if [[ "$experiment" != e40 ]]; then
    flock -w 60 "$root/runs/collect_${experiment}.lock" bash "$root/scripts/collect_rebuttal_stage.sh" "$experiment" \
      >>"$root/runs/collect_${experiment}_job${SLURM_JOB_ID}.log" 2>&1 || true
  fi
}
case "$worker" in
  0)
    cd "$root"
    export PYTHONPATH="/home/yanlongc/.e13_packages:$root"
    run_stage e38_seeds "$data/venv/bin/python" -m nar.e38_seed8b \
      --artifact-dir "$data/artifacts/e38/seed8b_e14_v1" --seeds 1 2 \
      --source-commit "$E38_SOURCE_COMMIT" --output "$root/results/e38/raw/seed8b_worker_job${SLURM_JOB_ID}.json"
    ;;
  1)
    code=/home/yanlongc/prismquant-rebuttal-e39
    cd "$code"
    export PYTHONPATH="/home/yanlongc/.prismquant/llama31-baseline-packages:/home/yanlongc/FlatQuant-upstream:/home/yanlongc/.e13_packages:$code"
    for phase in train-ppl tasks; do
      run_stage "e39_${phase}" "$data/venv/bin/python" -m nar.e39_flatquant \
        --checkpoint "$data/checkpoints/rebuttal_8b/flatquant_native_seed0" --phase "$phase" \
        --source-commit "$E39_SOURCE_COMMIT" --output "$code/results/e39/raw/native_8b_seed0_${phase}_job${SLURM_JOB_ID}.json"
    done
    ;;
  2)
    code=/home/yanlongc/prismquant-rebuttal-e40
    cd "$code"
    export PYTHONPATH="$code"
    export E40_CACHE_ROOT="$data/activations/llama31_8b/e40_matched_seed0_001"
    export E40_OUTPUT="$code/results/e40/run_job${SLURM_JOB_ID}"
    export E40_ARTIFACT_ROOT="$data/artifacts/e40_matched_seed0_001"
    run_stage e40_energy_ladder bash scripts/run_e40_pipeline.sh
    ;;
  3)
    code=/home/yanlongc/prismquant-rebuttal-e41
    cd "$code"
    export PYTHONPATH="$code"
    for group in 32 512 d; do
      run_stage "e41_g${group}" "$data/venv/bin/python" -m nar.e41_b6_mechanism \
        --group "$group" --seed 0 --source-commit "$E41_SOURCE_COMMIT" \
        --factor-output "$data/activations/llama31_8b/e41_b6/g${group}_seed0_v1" \
        --output "$code/results/e41/b6/raw/g${group}_seed0_job${SLURM_JOB_ID}.json"
    done
    ;;
  4)
    export E42_CODE_DIR=/home/yanlongc/prismquant-rebuttal-e42
    export E42_DUQUANT_ARTIFACT="$data/artifacts/e42/duquant_8b_seed0_v1"
    export E42_TASK_BATCH_SIZE=1
    for phase in quantize-ppl eight-tasks; do
      run_stage "e42_${phase}" env E42_DUQUANT_PHASE="$phase" bash "$E42_CODE_DIR/slurm_e42_duquant_native.sh"
    done
    ;;
  *) exit 2 ;;
esac
exit "$status"
