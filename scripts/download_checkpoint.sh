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

# Download from Hugging Face (or your hosting location)
# TODO: Replace with actual download URL
DOWNLOAD_URL="https://huggingface.co/ImmortalDemonGod/arc-taxonomy-reproduction/resolve/main/weights/champion-epoch=36-val_loss=0.5926.ckpt"

echo "Downloading from: $DOWNLOAD_URL"
echo ""

# Try wget first, fall back to curl
if command -v wget &> /dev/null; then
    wget -O "$CHECKPOINT_FILE" "$DOWNLOAD_URL"
elif command -v curl &> /dev/null; then
    curl -L -o "$CHECKPOINT_FILE" "$DOWNLOAD_URL"
else
    echo "❌ ERROR: Neither wget nor curl found!"
    echo "Please install wget or curl, or manually download from:"
    echo "$DOWNLOAD_URL"
    exit 1
fi

if [ -f "$CHECKPOINT_FILE" ]; then
    echo ""
    echo "✅ Download complete!"
    echo "   Location: $CHECKPOINT_FILE"
    echo "   Size: $(du -h "$CHECKPOINT_FILE" | cut -f1)"
else
    echo "❌ Download failed!"
    exit 1
fi
