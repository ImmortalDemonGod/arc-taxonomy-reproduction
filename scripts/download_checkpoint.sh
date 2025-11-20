#!/bin/bash
# Download Champion checkpoint for LoRA training
# This is the pretrained model from Trial 69 (champion_bootstrap.ckpt)

set -e

WEIGHTS_DIR="weights"
CHECKPOINT_FILE="$WEIGHTS_DIR/champion-epoch=36-val_loss=0.5926.ckpt"

# Create weights directory
mkdir -p "$WEIGHTS_DIR"

# Check if already exists
if [ -f "$CHECKPOINT_FILE" ]; then
    echo "✅ Checkpoint already exists: $CHECKPOINT_FILE"
    echo "   Size: $(du -h "$CHECKPOINT_FILE" | cut -f1)"
    exit 0
fi

echo "========================================================================"
echo "DOWNLOADING CHAMPION CHECKPOINT"
echo "========================================================================"
echo "File: champion-epoch=36-val_loss=0.5926.ckpt"
echo "Size: ~400MB"
echo "This is the pretrained Champion model from Trial 69"
echo ""

# NOTE: This checkpoint must be downloaded manually or uploaded to Hugging Face
# The file should be placed in: weights/champion-epoch=36-val_loss=0.5926.ckpt
# Size: ~21MB (not ~400MB - fixed in comment above)
echo "❌ ERROR: Checkpoint download not yet implemented"
echo ""
echo "Please download the checkpoint manually:"
echo "1. Download from: https://github.com/ImmortalDemonGod/arc-taxonomy-reproduction/releases"
echo "2. Save to: $CHECKPOINT_FILE"
echo "3. Expected size: ~21MB"
echo ""
echo "Or contact the authors for access to the pretrained weights."
exit 1

if [ -f "$CHECKPOINT_FILE" ]; then
    echo ""
    echo "✅ Download complete!"
    echo "   Location: $CHECKPOINT_FILE"
    echo "   Size: $(du -h "$CHECKPOINT_FILE" | cut -f1)"
else
    echo "❌ Download failed!"
    exit 1
fi
