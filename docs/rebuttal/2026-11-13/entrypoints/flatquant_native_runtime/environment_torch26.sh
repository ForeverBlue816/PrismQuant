# Isolated official-version runtime for A40 native FlatQuant; no shared pip install.
if type module >/dev/null 2>&1; then module load CUDA/12.4.0; fi
export CUDA_HOME=/cluster/apps/software/CUDA/12.4.0
export PATH="/projects/nar/nar-validation/e28-env/bin:/cluster/apps/software/CUDA/12.4.0/bin:$PATH"
export PYTHONPATH="/projects/nar/nar-validation/flatquant-native-runtime/packages:/home/yanlongc/.prismquant/llama31-baseline-packages:/home/yanlongc/.prismquant/FlatQuant-native-torch26-cu124:/home/yanlongc/.prismquant/FlatQuant-native-torch26-cu124/third-party/fast-hadamard-transform${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=2
export TMPDIR=/projects/nar/nar-validation/flatquant-native-runtime/tmp
