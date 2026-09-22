#!/bin/bash
#SBATCH -J BLG_plots
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=5gb
#SBATCH --time=72:0:0
#SBATCH --output=logs/%x/BLG_plots.out
#SBATCH --error=logs/%x/BLG_plots.err

cd "$SLURM_SUBMIT_DIR"

JOB_NAME="${SLURM_JOB_NAME:-BLG_plots}"
LOG_DIR="$SLURM_SUBMIT_DIR/logs/$JOB_NAME"
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
export MPLBACKEND=Agg  # brak wyświetlacza na węźle obliczeniowym

echo "[$(date '+%F %T')] Start job ${SLURM_JOB_ID:-nojob} on $(hostname)"
$SCRATCH/BLG_calc/bin/python -u -c "import sys; print('PYTHON_EXE=', sys.executable); print('PYTHON_VER=', sys.version); from mpi4py import MPI; import mumps; import kwant.solvers.mumps as km; import kwant.solvers.default as d; print('KWANT_SOLVER=', d.smodule.__name__); import runpy; runpy.run_path('main_BLG_plots.py', run_name='__main__')" 2>&1 | tee "$LOG_DIR/stdout.txt"
echo "[$(date '+%F %T')] End job ${SLURM_JOB_ID:-nojob}"