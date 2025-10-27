#!/bin/bash
# Regenerate distributional_alignment dataset with size-aware stratification
# This script safely removes old data and regenerates with 150 samples/task

set -e  # Exit on error

echo "========================================================================"
echo "DATA REGENERATION: distributional_alignment (60,000 examples)"
echo "========================================================================"
echo ""

DATA_DIR="data/distributional_alignment"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

# Check if metadata exists
if [ ! -f "$DATA_DIR/task_categories.json" ]; then
    echo "ERROR: task_categories.json not found!"
    echo "Make sure you're running from the reproduction package root."
    exit 1
fi

# Count existing task files
TASK_COUNT=$(find "$DATA_DIR" -name "*.json" -type f ! -name "task_categories.json" ! -name "split_manifest.json" ! -name "generation_statistics.json" 2>/dev/null | wc -l | tr -d ' ')

echo "Current state:"
echo "  Task files found: $TASK_COUNT"
echo "  Expected: 400 (with 150 samples each)"
echo ""

# Check if we need to regenerate
if [ "$TASK_COUNT" -eq 400 ]; then
    # Verify sample count in first task file
    FIRST_TASK=$(find "$DATA_DIR" -name "*.json" -type f ! -name "task_categories.json" ! -name "split_manifest.json" ! -name "generation_statistics.json" | head -1)
    if [ -n "$FIRST_TASK" ]; then
        SAMPLE_COUNT=$(python3 -c "import json; f=open('$FIRST_TASK'); d=json.load(f); print(len(d.get('train', [])))")
        echo "  Sample count in first task: $SAMPLE_COUNT"
        
        if [ "$SAMPLE_COUNT" -eq 150 ]; then
            echo ""
            echo "✅ Data is already up-to-date (400 tasks × 150 samples)"
            echo ""
            read -p "Regenerate anyway? (y/N): " -n 1 -r
            echo ""
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "Skipping regeneration."
                exit 0
            fi
        else
            echo "  ⚠️  OLD DATA DETECTED: Task has $SAMPLE_COUNT samples (expected 150)"
        fi
    fi
fi

# Remove old task files
echo ""
echo "🗑️  Removing old task files..."
find "$DATA_DIR" -name "*.json" -type f \
    ! -name "task_categories.json" \
    ! -name "split_manifest.json" \
    ! -name "generation_statistics.json" \
    -delete

echo "✅ Old data removed"
echo ""

# Regenerate data
echo "🔄 Generating 60,000 examples (400 tasks × 150 samples)..."
echo "⏱️  Estimated time: 15-20 minutes"
echo ""

python3 "$SCRIPT_DIR/generate_synthetic_arc_dataset.py" \
    --mode distributional_alignment \
    --samples-per-task 150 \
    --output-dir "$DATA_DIR"

# Verify generation
NEW_TASK_COUNT=$(find "$DATA_DIR" -name "*.json" -type f ! -name "task_categories.json" ! -name "split_manifest.json" ! -name "generation_statistics.json" | wc -l | tr -d ' ')

echo ""
echo "========================================================================"
echo "VERIFICATION"
echo "========================================================================"
echo "  Task files generated: $NEW_TASK_COUNT / 400"

if [ "$NEW_TASK_COUNT" -eq 400 ]; then
    echo "  Status: ✅ SUCCESS"
    echo ""
    echo "Data regeneration complete!"
    echo "Split manifest is already up-to-date with size-aware stratification."
    echo ""
    echo "Next steps:"
    echo "  python3 scripts/verify_split.py       # Verify split quality"
    echo "  ./run_training.sh champion            # Start training"
    exit 0
else
    echo "  Status: ❌ FAILED"
    echo ""
    echo "ERROR: Expected 400 task files, got $NEW_TASK_COUNT"
    echo "Check the generation log above for errors."
    exit 1
fi
