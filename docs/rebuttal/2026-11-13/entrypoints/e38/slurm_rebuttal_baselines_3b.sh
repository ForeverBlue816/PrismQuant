#!/bin/bash
#SBATCH --job-name=pq3b-flat-duquant
#SBATCH --partition=cluster02
#SBATCH --account=rose
#SBATCH --qos=rose
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --gpus=a40:2
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00
#SBATCH --no-requeue
#SBATCH --output=runs/rebuttal-baselines-3b-%j.out
#SBATCH --error=runs/rebuttal-baselines-3b-%j.err
set -uo pipefail
cd /home/yanlongc/prismquant-rebuttal-e38
: "${E39_SOURCE_COMMIT:?}" "${E42_SOURCE_COMMIT:?}"
unset SLURM_TRES_PER_TASK SLURM_GPUS SLURM_GPUS_PER_NODE SLURM_GPUS_PER_SOCKET SLURM_GPUS_PER_TASK
pids=()
for method in flatquant duquant; do
  srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=4 \
    --gpus-per-task=a40:1 --gpu-bind=single:1 --mem=39G \
    bash scripts/rebuttal_baseline_model_worker.sh llama32_3b "$method" accuracy \
    >"runs/baseline_3b_${method}_job${SLURM_JOB_ID}.out" \
    2>"runs/baseline_3b_${method}_job${SLURM_JOB_ID}.err" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
