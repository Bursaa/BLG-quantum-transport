#!/bin/bash
#SBATCH -J BLG_task8_chunk_TEST
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=5gb
#SBATCH --time=72:0:0
#SBATCH --output=logs/%x/BLG_data_%A_%a.out
#SBATCH --error=logs/%x/BLG_data_%A_%a.err

cd "$SLURM_SUBMIT_DIR"

JOB_NAME="${SLURM_JOB_NAME:-BLG_task8_chunk_TEST}"
export CHUNK_INDEX="${SLURM_ARRAY_TASK_ID:-${CHUNK_INDEX:-0}}"
LOG_DIR="$SLURM_SUBMIT_DIR/logs/$JOB_NAME/chunk_${CHUNK_INDEX}"
mkdir -p "$LOG_DIR"

module purge
module load GCC/14.3.0
module load OpenMPI/5.0.8
module load Python/3.11.5
module load MUMPS/5.8.1-metis
source "$SCRATCH/BLG_calc/bin/activate"

export OMPI_MCA_mtl=^ofi
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader,tcp
export LD_LIBRARY_PATH=/net/software/x86_64/el9/MUMPS/5.8.1-foss-2025b-metis/lib:${LD_LIBRARY_PATH}
export PYTHONUNBUFFERED=1

# Ustawienia domyślne chunkowania
export CHUNK_SIZE=${CHUNK_SIZE:-1}
export B_START=${B_START:-0.0}
export B_STOP=${B_STOP:-10.0}
export B_STEP=${B_STEP:-0.1}
export VB_START=${VB_START:-0.0}
export VB_STOP=${VB_STOP:-50.0}
export VB_STEP=${VB_STEP:-0.1}
export VT_VALUE=${VT_VALUE:-0.0}

echo "[$(date '+%F %T')] Start chunk ${CHUNK_INDEX} on $(hostname)"
echo "[$(date '+%F %T')] Chunk params: CHUNK_SIZE=${CHUNK_SIZE}, B=[${B_START},${B_STOP}] step=${B_STEP}, Vb=[${VB_START},${VB_STOP}] step=${VB_STEP}, Vt=${VT_VALUE}"
$SCRATCH/BLG_calc/bin/python -u -c "import sys; print('PYTHON_EXE=', sys.executable); print('PYTHON_VER=', sys.version); from mpi4py import MPI; import mumps; import kwant.solvers.mumps as km; import kwant.solvers.default as d; print('KWANT_SOLVER=', d.smodule.__name__); import runpy; runpy.run_path('main_BLG_task8_chunk.py', run_name='__main__')" 2>&1 | tee "$LOG_DIR/stdout_${CHUNK_INDEX}.txt"
echo "[$(date '+%F %T')] End chunk ${CHUNK_INDEX}"
