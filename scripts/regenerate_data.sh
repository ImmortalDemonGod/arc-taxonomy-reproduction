#!/bin/bash
# Regenerate distributional_alignment dataset with size-aware stratification
# Usage: ./regenerate_data.sh [samples_per_task]
# Default: 150 samples/task (Phase 0 - visual classifier)
# Alternative: 400 samples/task (Phase 1A-v2 - test Neural Affinity on all 400 tasks)

set -e  # Exit on error

# Parse command-line arguments
SAMPLES_PER_TASK=${1:-150}  # Default to 150 if not specified

TOTAL_EXAMPLES=$((400 * SAMPLES_PER_TASK))

echo "========================================================================"
echo "DATA REGENERATION: distributional_alignment"
echo "========================================================================"
echo "  Samples per task: $SAMPLES_PER_TASK"
echo "  Total examples: $TOTAL_EXAMPLES (400 tasks × $SAMPLES_PER_TASK samples)"
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
echo "  Expected: 400 (with $SAMPLES_PER_TASK samples each)"
echo ""

# Check if we need to regenerate
if [ "$TASK_COUNT" -eq 400 ]; then
    # Verify sample count in first task file
    FIRST_TASK=$(find "$DATA_DIR" -name "*.json" -type f ! -name "task_categories.json" ! -name "split_manifest.json" ! -name "generation_statistics.json" | head -1)
    if [ -n "$FIRST_TASK" ]; then
        SAMPLE_COUNT=$(python3 -c "import json; f=open('$FIRST_TASK'); d=json.load(f); print(len(d.get('train', [])))")
        echo "  Sample count in first task: $SAMPLE_COUNT"
        
        if [ "$SAMPLE_COUNT" -eq "$SAMPLES_PER_TASK" ]; then
            echo ""
            echo "✅ Data is already up-to-date (400 tasks × $SAMPLES_PER_TASK samples)"
            echo ""
            read -p "Regenerate anyway? (y/N): " -n 1 -r
            echo ""
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "Skipping regeneration."
                exit 0
            fi
        else
            echo "  ⚠️  OLD DATA DETECTED: Task has $SAMPLE_COUNT samples (expected $SAMPLES_PER_TASK)"
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
ESTIMATED_MINUTES=$(( (SAMPLES_PER_TASK * 400) / 4000 ))  # Rough estimate: ~4k samples/min
echo "🔄 Generating $TOTAL_EXAMPLES examples (400 tasks × $SAMPLES_PER_TASK samples)..."
echo "⏱️  Estimated time: $ESTIMATED_MINUTES-$(($ESTIMATED_MINUTES + 5)) minutes"
echo ""

python3 "$SCRIPT_DIR/generate_synthetic_arc_dataset.py" \
    --mode distributional_alignment \
    --samples-per-task $SAMPLES_PER_TASK \
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
    
    # Create size-aware train/val split
    echo "========================================================================"
    echo "CREATING SIZE-AWARE SPLIT"
    echo "========================================================================"
    python3 "$SCRIPT_DIR/create_size_aware_split.py"
    
    if [ $? -ne 0 ]; then
        echo "❌ ERROR: Split creation failed"
        exit 1
    fi
    
    echo ""
    echo "========================================================================"
    echo "VERIFYING SPLIT QUALITY"
    echo "========================================================================"
    python3 "$SCRIPT_DIR/verify_split.py"
    
    if [ $? -ne 0 ]; then
        echo "❌ WARNING: Split verification had issues (but continuing)"
    fi
    
    echo ""
    echo "========================================================================"
    echo "✅ DATA REGENERATION COMPLETE"
    echo "========================================================================"
    echo "Generated: 400 tasks × $SAMPLES_PER_TASK samples = $TOTAL_EXAMPLES examples"
    echo "Split created: train/val with size-aware stratification"
    echo ""
    if [ "$SAMPLES_PER_TASK" -eq 150 ]; then
        echo "Mode: Phase 0 (visual classifier - category centroids)"
    else
        echo "Mode: Phase 1A-v2 (test Neural Affinity across all categories)"
    fi
    echo ""
    echo "Next steps:"
    echo "  ./run_training.sh test      # Quick smoke test"
    echo "  ./run_training.sh champion  # Full training"
    exit 0
else
    echo "  Status: ❌ FAILED"
    echo ""
    echo "ERROR: Expected 400 task files, got $NEW_TASK_COUNT"
    echo "Check the generation log above for errors."
    exit 1
fi
