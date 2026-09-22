#!/bin/bash
# Submituje oddzielny job Slurm dla każdej wartości B.
# Edytuj B_VALUES, VT_VALUE, VB_VALUE przed uruchomieniem.

# d=60nm
# B_VALUES=(0.49 0.75 1.1 2.2 2.9 3.45 5.0 6.25 7.22 7.7 8.98 9.6) #Vb=20
# B_VALUES=(0.48 0.7 1.3 2.18 4.98 5.75 6.3 6.8 7.8 8.85) #Vb=30
# B_VALUES=(0.49 0.7 1.22 2.0 2.9 3.85 4.85 5.75 7.6 9.2) #Vb=45

# d=0.01
# B_VALUES=(0.10 0.70 2.11 2.91 3.75 6.23 7.71 8.46 9.2 0.34 1.20 2.65 3.15 4.93 7.31 8.18 8.72 9.82) #Vb=20
# B_VALUES=(0.08 0.62 2.11 3.85 5.99 7.52  9.16 0.32 1.22 2.95 4.75 7.25 7.74 9.82) #Vb=30
# B_VALUES=(0.30 1.30 4.51 5.81 6.95 0.10 0.62 3.05 5.43 6.15 8.32) #Vb=45


VT_VALUE=0.0
VB_VALUE=20.0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for B_T in "${B_VALUES[@]}"; do
    JOB_NAME="BLG_task9_Vb${VB_VALUE}_B${B_T}_R_to_L"
    LOG_DIR="$SCRIPT_DIR/logs/${JOB_NAME}"
    mkdir -p "$LOG_DIR"

    sbatch \
        --job-name="$JOB_NAME" \
        --export=ALL,B_T="$B_T",VT_VALUE="$VT_VALUE",VB_VALUE="$VB_VALUE" \
        "$SCRIPT_DIR/run_BLG_task9.sh"

    echo "Submitted B_T=${B_T}"
done
