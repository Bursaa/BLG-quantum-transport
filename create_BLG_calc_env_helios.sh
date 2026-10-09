#!/bin/bash
# Run on a COMPUTE node (login node has an incompatible GLIBC):
#   srun --pty --mem=8gb --time=1:0:0 bash
#   bash create_BLG_calc_env_helios.sh

module purge
module load GCC/14.3.0
module load OpenMPI/5.0.8
module load Python/3.11.5
module load MUMPS/5.8.1-metis

python -m venv $SCRATCH/BLG_calc
source $SCRATCH/BLG_calc/bin/activate

pip install --upgrade pip
pip install numpy scipy matplotlib

MPICC=mpicc pip install mpi4py
pip install --force-reinstall --no-binary kwant --no-cache-dir kwant
MPICC=mpicc pip install --no-binary mumps --no-cache-dir mumps

export LD_LIBRARY_PATH=/net/software/x86_64/el9/MUMPS/5.8.1-foss-2025b-metis/lib:${LD_LIBRARY_PATH}
export OMPI_MCA_mtl=^ofi
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader,tcp

python -c "from mpi4py import MPI; import mumps; import kwant.solvers.mumps as km; import kwant.solvers.default as d; print('KWANT_SOLVER=', d.smodule.__name__)"

