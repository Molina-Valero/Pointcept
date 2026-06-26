#!/bin/bash
#PBS -N install_pointcept
#PBS -l select=1:ncpus=4:mem=32gb:scratch_local=50gb:ngpus=1:gpu_cap=cuda80:cuda_version=12.4
#PBS -l walltime=4:00:00
#PBS -m ae

# ============================================================
# EDIT THIS: change brno2 to your actual storage location
# Check yours with: echo $HOME  after logging in
# ============================================================
STORAGE=/storage/brno2/home/molina_valero
ENV_PREFIX=$STORAGE/envs/pointcept-torch2.5.0-cu12.4
WORKDIR=/storage/brno2/home/molina_valero/Pointcept

cd $WORKDIR || exit 1

# Redirect temp files to scratch to avoid quota issues
export TMPDIR=$SCRATCHDIR
export PIP_CACHE_DIR=$SCRATCHDIR/pip_cache

# Load mambaforge (recommended over conda-modules on MetaCentrum)
module add mambaforge

# Create the environment on persistent storage
mamba env create --prefix $ENV_PREFIX -f environment.yml

echo "================================================"
echo "Environment created at: $ENV_PREFIX"
echo ""
echo "To activate in future jobs, add these lines:"
echo "  module add mambaforge"
echo "  mamba activate $ENV_PREFIX"
echo "================================================"
