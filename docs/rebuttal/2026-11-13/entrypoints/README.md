# Archived execution entrypoints

These files preserve the cluster execution paths used for the reported or queued experiments. They require the original worktrees, checkpoints, datasets and pinned environments; this directory is not a standalone reproduction package.

- [E38 original/optimized deployment runner](e38/quarot-llama3/e38/stage_a_8b.py) and [rotation implementation](e38/quarot-llama3/e38/optimized_rotation.py).
- [Shared baseline timing protocol](e38/scripts/baseline_timing.py) and [same-GPU comparison job](e38/slurm_rebuttal_speed_8b.sh).
- [FlatQuant official recipe runner](e39/nar/e39_flatquant.py), [native export/measurement runner](e39/nar/e39_flatquant_native.py), and [auditable dead-allocation removal](e39/nar/e39_native_cache_optimization.py).
- [DuQuant official recipe runner](e42/nar/e42_duquant_native.py) and [Transformers/cache compatibility adapter](e42/nar/e42_duquant_compat.py).
- [FlatQuant native environment](flatquant_native_runtime/environment_torch26.sh) and [runtime provenance](flatquant_native_runtime/native_runtime_manifest.json).

See [manifest.json](../manifest.json) for source commits and per-file hashes, and the [snapshot index](../README.md) for actual completion status. No numeric gate is used to suppress poor-quality timing results. GPU exclusivity and valid incremental caches are structural requirements for a valid speed measurement.
