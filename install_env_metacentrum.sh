#!/bin/bash
#PBS -N install_pointcept
#PBS -l select=1:ncpus=4:mem=64gb:scratch_local=50gb:ngpus=1
#PBS -l walltime=6:00:00
#PBS -m ae

# ============================================================
# EDIT THIS: change brno2 to your actual storage location
# Check yours with: echo $HOME  after logging in
# ============================================================
STORAGE=/storage/brno2/home/molina_valero
ENV_PREFIX=$STORAGE/envs/pointcept
WORKDIR=/storage/brno2/home/molina_valero/Pointcept

cd $WORKDIR || exit 1

# Redirect temp files to scratch to avoid quota issues
export TMPDIR=$SCRATCHDIR
export PIP_CACHE_DIR=$SCRATCHDIR/pip_cache

# Load mambaforge (recommended over conda-modules on MetaCentrum)
module add mambaforge

# Create the environment on persistent storage
mamba env create --prefix $ENV_PREFIX -f environment_metacentrum.yml --force

# -------------------------------------------------------
# Post-create: install packages that need --no-build-isolation
# because they import torch/setuptools at build time and need
# the already-installed torch + cuda-toolkit to compile against.
# -------------------------------------------------------

# flash-attention: imports torch at build time, same issue as pointops
mamba run --prefix $ENV_PREFIX \
    pip install flash-attn --no-build-isolation

mamba run --prefix $ENV_PREFIX \
    pip install --no-build-isolation ./libs/pointops

mamba run --prefix $ENV_PREFIX \
    pip install --no-build-isolation ./libs/pointops2

mamba run --prefix $ENV_PREFIX \
    pip install --no-build-isolation ./libs/pointrope

mamba run --prefix $ENV_PREFIX \
    pip install --no-build-isolation ./libs/pointgroup_ops

echo "================================================"
echo "Environment created at: $ENV_PREFIX"
echo ""
echo "To activate in future jobs, add these lines:"
echo "  module add mambaforge"
echo "  mamba activate $ENV_PREFIX"
echo "================================================"
