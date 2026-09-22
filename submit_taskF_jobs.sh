#!/bin/bash
# Submituje oddzielny job Slurm dla każdej wartości B — zadanie F (dyspersja).

B_VALUES=(1.670 1.89 2.0 2.355 2.49 2.72 3.10 3.6 4.96 5.85 7.4 7.63 7.71 7.92 8.28)
VT_VALUE=0.0
VB_VALUE=30.0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for B_T in "${B_VALUES[@]}"; do
    JOB_NAME="BLG_taskF_B${B_T}"
    LOG_DIR="$SCRIPT_DIR/logs/${JOB_NAME}"
    mkdir -p "$LOG_DIR"

    sbatch \
        --job-name="$JOB_NAME" \
        --export=ALL,B_T="$B_T",VT_VALUE="$VT_VALUE",VB_VALUE="$VB_VALUE" \
        "$SCRIPT_DIR/run_BLG_task_F.sh"

    echo "Submitted B_T=${B_T}"
done
