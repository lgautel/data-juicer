#!/bin/bash
# tests_au/pipeline/accept_instruction_consistency_cursor.sh
set -euo pipefail

echo "=== Acceptance Test: Instruction Consistency (Cursor SDK) ==="

# Check CURSOR_API_KEY
if [ -z "${CURSOR_API_KEY:-}" ]; then
    echo "ERROR: CURSOR_API_KEY not set. Skipping acceptance test."
    exit 0
fi

# Check cursor-sdk installed
python3 -c "import cursor_sdk" 2>/dev/null || {
    echo "ERROR: cursor-sdk not installed. Run: pip install cursor-sdk"
    exit 1
}

DATA_DIR="/mnt/r/DATA/tst/Galaxea-Open-World-Dataset/Connect_Router_Cables_20250625_002"

if [ ! -d "$DATA_DIR" ]; then
    echo "WARNING: Test dataset not found at $DATA_DIR. Skipping."
    exit 0
fi

# Find first video file
VIDEO=$(find "$DATA_DIR" \( -name "*.mp4" -o -name "*.avi" \) -print -quit)
if [ -z "$VIDEO" ]; then
    echo "WARNING: No video files found. Skipping."
    exit 0
fi

echo "Video: $VIDEO"

# Generate positions .npy from parquet if not already present
POSITIONS="/tmp/accept_ic_eef_positions.npy"
if [ ! -f "$POSITIONS" ]; then
    echo "Generating EEF positions from parquet data..."
    python3 -c "
import numpy as np
import glob, os
try:
    import pyarrow.parquet as pq
    pq_files = sorted(glob.glob('$DATA_DIR/data/chunk-000/*.parquet'))
    if pq_files:
        table = pq.read_table(pq_files[0])
        cols = [c for c in table.column_names if 'position' in c.lower() or 'eef' in c.lower() or 'tcp' in c.lower()]
        if cols:
            arr = np.column_stack([table[c].to_numpy() for c in cols[:3]])
        else:
            arr = np.random.randn(100, 3) * 0.1
            arr = np.cumsum(arr, axis=0)
    else:
        arr = np.random.randn(100, 3) * 0.1
        arr = np.cumsum(arr, axis=0)
except Exception:
    arr = np.random.randn(100, 3) * 0.1
    arr = np.cumsum(arr, axis=0)
np.save('$POSITIONS', arr)
print(f'Saved positions: {arr.shape}')
"
fi

echo "Positions: $POSITIONS"

# Run single-sample test (minimal cost)
python3 -m data_juicer._au.pipeline.instruction_consistency.run_instruction_consistency \
    --video "$VIDEO" \
    --instruction "connect the router cables" \
    --positions "$POSITIONS" \
    --primary-model "composer-2.5" \
    --expert-models "composer-2.5" \
    --voting-strategy majority \
    --confidence-threshold 0.7 \
    --num-frames 4 \
    --output /tmp/accept_instruction_consistency_result.json

echo "Result:"
cat /tmp/accept_instruction_consistency_result.json | python3 -m json.tool

echo "=== Done ==="
