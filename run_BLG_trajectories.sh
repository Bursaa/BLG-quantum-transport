#!/bin/bash
#SBATCH -J BLG_trajectories_d60_l_to_r
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=4gb
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

cd "$SLURM_SUBMIT_DIR"
module purge
module load GCC/14.3.0
module load OpenMPI/5.0.8
module load Python/3.11.5
source "$SCRATCH/BLG_calc/bin/activate"
export PYTHONUNBUFFERED=1

python3 -u main_BLG_trajectories.py \
  --vt 0.0 \
  --d-nm 60.0 \
  --x-limit-nm 200.0 \
  --y-limit-nm 150.0 \
  --out-dir data_d=60_AB/trajectories_l_to_r
