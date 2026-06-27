#!/bin/bash
#PBS -N install_pointcept
#PBS -l select=1:ncpus=4:mem=32gb:scratch_local=50gb:ngpus=1:gpu_mem=16gb:gpu_cap=compute_80
#PBS -l walltime=6:00:00
#PBS -m ae

STORAGE=/storage/brno2/home/molina_valero
ENV_PREFIX=$STORAGE/envs/pointcept-torch2.7.0-cu12.6
WORKDIR=/storage/brno2/home/molina_valero/Pointcept

cd $WORKDIR || exit 1

# Redirect temp files to scratch to avoid quota issues
export TMPDIR=$SCRATCHDIR
export PIP_CACHE_DIR=$SCRATCHDIR/pip_cache

# Load mambaforge
module add mambaforge

# Show system CUDA version for reference
echo "System nvcc: $(nvcc --version 2>/dev/null || echo 'not found')"
echo "System CUDA: $(nvidia-smi | grep 'CUDA Version' || echo 'n/a')"

# Create the base environment (without pointops/pointgroup_ops)
mamba env create --prefix $ENV_PREFIX -f environment_metacentrum.yml

# Now build pointops and pointgroup_ops using the CUDA bundled inside the env,
# so we are never affected by the system CUDA version
echo "=== Building CUDA extensions ==="
source activate $ENV_PREFIX

# Force use of the conda env's own CUDA toolkit
export CUDA_HOME=$ENV_PREFIX
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "nvcc in env: $(nvcc --version)"

cd $WORKDIR/libs/pointops
pip install -e .

cd $WORKDIR/libs/pointgroup_ops
pip install -e .

echo "================================================"
echo "Environment created at: $ENV_PREFIX"
echo ""
echo "Verify with:"
echo "  python -c \"import pointops; print('pointops OK')\""
echo ""
echo "To activate in future jobs:"
echo "  module add mambaforge"
echo "  source activate $ENV_PREFIX"
echo "================================================"

clean_scratch
