#!/bin/bash
# Standalone submitter: stałe B, zmienne Vb.
# Nie modyfikuje starego submit_task9_jobs.sh.
# Ustaw B_VALUE, VT_VALUE i VB_VALUES przed uruchomieniem.

B_VALUE=3.5
VT_VALUE=0.0

#d=60
# VB_VALUES=(12.80 16.80 21.00 23.40 25.70 28.40 33.80 37.50 40.40 46.00 48.607.10 15.50 17.80 22.40 24.50 27.00 30.10 35.70 38.90 44.50 47.20)

#d=0
VB_VALUES=(5.40 8.10 12.50 14.60 17.40 19.70 22.60 25.70 29.40 32.90 36.00 40.00 43.50 47.40 1.10 6.90 9.30 13.70 16.10 18.50 21.20 24.00 28.20 30.90 34.80 37.40 41.80 45.40)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for VB_VALUE in "${VB_VALUES[@]}"; do
    JOB_NAME="BLG_task9_B${B_VALUE}_Vb${VB_VALUE}_R_to_L"
    LOG_DIR="$SCRIPT_DIR/logs/${JOB_NAME}"
    mkdir -p "$LOG_DIR"

    sbatch \
        --job-name="$JOB_NAME" \
        --export=ALL,B_T="$B_VALUE",VT_VALUE="$VT_VALUE",VB_VALUE="$VB_VALUE" \
        "$SCRIPT_DIR/run_BLG_task9.sh"

    echo "Submitted Vb=${VB_VALUE} at B=${B_VALUE} T"
done
