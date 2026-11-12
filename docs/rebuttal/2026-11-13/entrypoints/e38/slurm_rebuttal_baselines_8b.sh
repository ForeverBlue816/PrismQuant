#!/bin/bash
#SBATCH --job-name=pq8b-flat-duquant
#SBATCH --partition=cluster02
#SBATCH --account=rose
#SBATCH --qos=rose
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --gpus=pro6000:2
#SBATCH --mem=66G
#SBATCH --time=3-00:00:00
#SBATCH --no-requeue
#SBATCH --output=runs/rebuttal-baselines-8b-%j.out
#SBATCH --error=runs/rebuttal-baselines-8b-%j.err
# Latest scope: official FlatQuant and DuQuant on Meta-Llama-3.1-8B only.
set -uo pipefail
cd /home/yanlongc/prismquant-rebuttal-e38
: "${E39_SOURCE_COMMIT:?}" "${E42_SOURCE_COMMIT:?}"
mkdir -p runs
unset SLURM_TRES_PER_TASK SLURM_GPUS SLURM_GPUS_PER_NODE SLURM_GPUS_PER_SOCKET SLURM_GPUS_PER_TASK
pids=()
# Existing worker1 is FlatQuant train/PPL/tasks; worker4 is DuQuant quant/PPL/tasks.
for worker in 1 4; do
  srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=4 \
    --gpus-per-task=pro6000:1 --gpu-bind=single:1 --mem=33G \
    bash scripts/rebuttal_compute_worker.sh "$worker" \
    >"runs/baseline_worker${worker}_job${SLURM_JOB_ID}.out" \
    2>"runs/baseline_worker${worker}_job${SLURM_JOB_ID}.err" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
