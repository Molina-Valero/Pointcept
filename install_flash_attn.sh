#!/bin/bash
#PBS -N install_flash_attn
#PBS -l select=1:ncpus=4:mem=64gb:scratch_local=20gb:ngpus=1
#PBS -l walltime=4:00:00
#PBS -m ae

set -euo pipefail

STORAGE=/storage/brno2/home/molina_valero
ENV_PREFIX=$STORAGE/envs/pointcept

# Redirect temp/cache to scratch, same reasoning as the main install script
export TMPDIR=$SCRATCHDIR
export PIP_CACHE_DIR=$SCRATCHDIR/pip_cache
mkdir -p "$PIP_CACHE_DIR"

export CONDA_PKGS_DIRS=$SCRATCHDIR/conda_pkgs
mkdir -p "$CONDA_PKGS_DIRS"

# flash-attn compiles CUDA kernels per-arch and can be heavy; keep parallel
# nvcc jobs capped so it doesn't blow past requested memory.
export MAX_JOBS=2

module add mambaforge

mamba run --prefix $ENV_PREFIX \
    pip install flash-attn --no-build-isolation

echo "flash-attn installed into $ENV_PREFIX"
