#!/bin/bash
#SBATCH --job-name=pq-baselines-timing
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
#SBATCH --output=runs/rebuttal-baselines-timing-%j.out
#SBATCH --error=runs/rebuttal-baselines-timing-%j.err
set -uo pipefail
cd /home/yanlongc/prismquant-rebuttal-e38
: "${BASELINE_MODEL:?}" "${E39_SOURCE_COMMIT:?}" "${E42_SOURCE_COMMIT:?}"
status=0
# Both real quantized models run sequentially on the same exclusive GPU.
# Each worker loads its saved checkpoint once for full PPL then cached timing.
for method in flatquant duquant; do
  bash scripts/rebuttal_baseline_model_worker.sh "$BASELINE_MODEL" "$method" timing || status=1
done
exit "$status"
