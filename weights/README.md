# Pre-trained Model Weights

**Status:** To be uploaded in Week 1-2

---

## Download Instructions

### Option 1: Direct Download
```bash
# Download from [URL to be provided]
wget [URL]/pretrained_model.pt

# Verify checksum
sha256sum pretrained_model.pt
# Expected: [checksum to be provided]
```

### Option 2: Hugging Face Hub
```bash
# Install huggingface_hub
pip install huggingface_hub

# Download
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='[repo]', filename='pretrained_model.pt', local_dir='.')"
```

---

## Model Details

**Architecture:** Simplified Transformer  
**Parameters:** ~[X]M  
**Training Data:** re-arc training set  
**Training Duration:** [X] hours on [GPU type]

**Performance:**
- Pre-training accuracy: [X]%
- Average fine-tuning improvement: [X]%

---

## File Structure

```
weights/
├── README.md (this file)
├── pretrained_model.pt (to be downloaded)
└── .gitkeep
```

---

## Checksum Verification

After downloading, verify the file integrity:

```bash
sha256sum pretrained_model.pt
```

Expected checksum: `[to be provided]`

---

## Usage

```python
from model import ARCTransformer

# Load pre-trained model
model = ARCTransformer.load_pretrained('weights/pretrained_model.pt')
```

---

**Note:** Weights will be uploaded after model extraction in Week 1-2.
