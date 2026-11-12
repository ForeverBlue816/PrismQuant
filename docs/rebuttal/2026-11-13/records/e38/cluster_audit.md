> Historical record snapshot. Some original job states and plans are stale. Use the [current status index](../../README.md) for the active scope and observed scheduler state.

# E38 cluster and environment audit

This is a read-only inventory from 10 November 2026 (Singapore time), not an accuracy or performance result. The timestamped commands, scheduler output, import results and library versions are preserved in [raw/cluster_audit.json](raw/cluster_audit.json). No GPU job was submitted, cancelled or used by this audit.

## Scheduler resources

The `cluster02` partition exposes 90 `pro6000` GPUs, with the `gpu_96g` feature, and 18 NVIDIA A40 GPUs across `gpu-a40-1` (10) and `gpu-a40-2` (8). All GPU resources were allocated at observation. The `MIXED` status on `gpu-l40-1` represents available CPU cores; its four GPUs were allocated. The 96 GB classification comes from Slurm features and must be confirmed with `nvidia-smi` inside the eventual allocation.

The user association is account `rose`, with default QOS `rose` and permission to use `override-limits-but-killable`. QOS `rose` allows up to eight `pro6000` GPUs and eight A40 GPUs, has a maximum of two running jobs and five submitted jobs, and can preempt the override QOS. The override QOS allows eight GPUs in total and five submitted jobs, with requeue on preemption. At observation one running user job used `rose`, while the CPU driver used the override QOS; the two jobs must not be combined when assessing the `rose` running-job cap. Existing jobs were left untouched.

Historical Slurm accounting confirms E14 job `141650` completed with one `pro6000` GPU, four CPU cores and 90 GiB host memory on `gpu-pro6000-9`, using account/QOS `rose/rose`. A40 timing jobs `143654` and `143743` also completed. A40 is therefore reachable through the scheduler, although an idle allocation was not available during this audit.

## Environments

| Component | E14 environment | Existing E28 deployment environment |
|---|---|---|
| Python executable | `/projects/nar/nar-validation/venv/bin/python` | `/projects/nar/nar-validation/e28-env/bin/python` |
| Python | 3.11.16 | 3.11.16 |
| PyTorch | 2.11.0+cu128 | 2.4.1+cu124 |
| Transformers | 5.16.0.dev0 | 4.36.2 |
| Triton | 3.6.0 | 3.0.0 |
| datasets | 5.0.1 | 2.17.1 |
| accelerate | 1.14.0 | 0.27.2 |

The E14 environment matches the research checkout's saved requirements. Its Transformers source commit is `a353632607c59463e6ced86a44c2de3c2cd62d5e`. The harness checkout is clean at `b954108c9baaaa934b4ad842033b31a97ee30816`; the local harness installation resolves from `/home/yanlongc/.e13_packages`. There is no established environment drift in this audit. The main packages resolve under `/projects/river/envs/qwen35/lib/python3.11/site-packages` because the E14 virtual environment includes system site packages.

The CPU import check succeeded for `torch`, `transformers`, `triton`, `datasets`, `accelerate`, `lm_eval`, and `nar.e14_w4a4kv4`. The E28 environment remains separate and its historical protocol specifies no dependency upgrade. Connecting the E14 and E28 paths therefore requires explicit compatibility/equivalence checks; importing the E14 module alone does not establish model equivalence.

## Bounded first allocation

The prepared `slurm_e38_anchor.sh` uses the successful historical resource envelope, bounded to two hours: one `pro6000` GPU, `highmem`, four CPU cores, 90 GiB host memory, `cluster02`, account/QOS `rose/rose`. Pending resource availability is expected. This BF16 anchor needs no timing exclusivity because it measures accuracy only.

Inside the allocation, the run must preserve GPU model/UUID/driver and memory inventory, code commit and source hashes, exact model/token hashes, command line, seed, library versions and failure details. Formal E38 timings remain conditional on an idle exclusive GPU and sufficiently quiet host, with all comparison rows measured on the same GPU model. No timing table is supplied by this audit.
