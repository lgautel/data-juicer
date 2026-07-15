#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VENV="${DJ_VENV:-/mnt/r/VENV/dj}"
DATASET="/mnt/r/DATA/tst/Galaxea-Open-World-Dataset/Connect_Router_Cables_20250625_002"
OUTPUT_DIR="$SCRIPT_DIR/outputs"

if [ -x "$VENV/bin/python" ]; then
    PYTHON="$VENV/bin/python"
    DJ_PROCESS="$VENV/bin/dj-process"
else
    PYTHON="$(which python3)"
    DJ_PROCESS="$(which dj-process)"
fi

cd "$REPO_ROOT"

echo "=== Step 1: Convert LeRobot episodes to JSONL ==="
EPISODES_JSONL="$OUTPUT_DIR/lerobot_episodes.jsonl"
if [ ! -f "$EPISODES_JSONL" ]; then
    "$PYTHON" "$SCRIPT_DIR/convert_lerobot_episodes.py" \
        --dataset_dir "$DATASET" \
        --output "$EPISODES_JSONL"
    echo "Converted $(wc -l < "$EPISODES_JSONL") episodes."
else
    echo "Episodes JSONL already exists ($(wc -l < "$EPISODES_JSONL") lines), skipping conversion."
fi

echo ""
echo "=== Step 2: Run Check 3 pipeline ==="
"$DJ_PROCESS" --config "$SCRIPT_DIR/accept_video_quality_filtering.yaml"

echo ""
echo "=== Step 3: Verify results ==="
"$PYTHON" <<'PYEOF'
import json
import sys

result_path = "tests_au/ops/filter/outputs/check3_result.jsonl"
stats_path  = "tests_au/ops/filter/outputs/check3_stats.jsonl"

with open(result_path) as f:
    results = [json.loads(line) for line in f]

print(f"Result episodes: {len(results)}")

META_KEYS = [
    "key_frame_report",
    "static_segment_report",
    "video_quality_removal_mask",
    "frame_removal_safety_report",
    "frame_removal_report",
]
STATS_KEYS = [
    "frame_removal_safe",
    "frame_removal_max_discontinuity",
    "frame_removal_removed_ratio",
    "frame_removal_remaining_frames",
]

errors = []
for i, row in enumerate(results):
    meta = row.get("__dj__meta__", {})
    stats = row.get("__dj__stats__", {})
    ep_id = row.get("id", f"row_{i}")

    for mk in META_KEYS:
        if mk not in meta:
            errors.append(f"{ep_id}: missing meta key '{mk}'")

    for sk in STATS_KEYS:
        if sk not in stats:
            errors.append(f"{ep_id}: missing stats key '{sk}'")

    # Check frame removal report consistency
    fr_report_str = meta.get("frame_removal_report")
    if fr_report_str:
        fr_report = json.loads(fr_report_str) if isinstance(fr_report_str, str) else fr_report_str
        if not fr_report.get("skipped", False):
            remaining = fr_report.get("remaining_frames", 0)
            actual_states_len = len(row.get("states", []))
            if actual_states_len != remaining:
                errors.append(
                    f"{ep_id}: states length {actual_states_len} != "
                    f"reported remaining {remaining}"
                )

    # Print per-episode summary
    safe = stats.get("frame_removal_safe", "N/A")
    ratio = stats.get("frame_removal_removed_ratio", 0)
    max_disc = stats.get("frame_removal_max_discontinuity", 0)
    print(f"  {ep_id}: safe={safe}, removed_ratio={ratio:.3f}, max_disc={max_disc:.4f}")

if errors:
    print(f"\n*** {len(errors)} ERRORS ***")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print(f"\nAll {len(results)} episodes verified.")
    print("ACCEPTANCE PASSED")
PYEOF
