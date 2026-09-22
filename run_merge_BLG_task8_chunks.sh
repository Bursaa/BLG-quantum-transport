#!/bin/bash
#SBATCH -J BLG_task8_merge_TEST
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=5gb
#SBATCH --time=72:0:0
#SBATCH --output=logs/%x/BLG_data.out
#SBATCH --error=logs/%x/BLG_data.err

cd "$SLURM_SUBMIT_DIR"

JOB_NAME="${SLURM_JOB_NAME:-BLG_task8_merge_TEST}"
LOG_DIR="$SLURM_SUBMIT_DIR/logs/$JOB_NAME"
mkdir -p "$LOG_DIR"

module purge
module load GCC/14.3.0
module load OpenMPI/5.0.8
module load Python/3.11.5
module load MUMPS/5.8.1-metis
source "$SCRATCH/BLG_calc/bin/activate"

export PYTHONUNBUFFERED=1

echo "[$(date '+%F %T')] Start merge"
$SCRATCH/BLG_calc/bin/python -u merge_BLG_task8_chunks.py 2>&1 | tee "$LOG_DIR/stdout.txt"
echo "[$(date '+%F %T')] End merge"
