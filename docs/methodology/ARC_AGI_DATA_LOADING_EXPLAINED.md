# ARC-AGI-2 Data Loading: Design & Implementation

**Date:** November 1, 2025  
**Purpose:** Document how ARC-AGI-2 data loading works and differs from re-arc

---

## 1. Data Format Comparison

### **Re-ARC (Synthetic) Format**
```
Location: data/distributional_alignment/
Structure: 400 individual JSON files (one per task)
Naming: {task_id}.json (e.g., "007bbfb7.json")

File Structure:
{
  "train": [
    {"input": [[...]], "output": [[...]]}  // 150 examples
  ],
  "test": []  // Usually empty (all examples in train)
}
```

### **ARC-AGI-2 (Real Competition) Format**
```
Location: data/arc_prize_2025/
Structure: 2 large JSON files (all tasks in one file)
Files:
  - arc-agi_training_challenges.json (1000 tasks)
  - arc-agi_evaluation_challenges.json (120 tasks)

File Structure:
{
  "task_id_1": {
    "train": [
      {"input": [[...]], "output": [[...]]}  // 2-5 examples
    ],
    "test": [
      {"input": [[...]]}  // NO OUTPUT (for prediction)
    ]
  },
  "task_id_2": {...},
  ...
}
```

**CRITICAL DIFFERENCE:** ARC-AGI-2 test examples don't have outputs!

---

## 2. Data Loading Implementation

### **Step 1: File Format Conversion** (train_exp3_champion.py lines 96-123)

ARC-AGI-2 comes as dict-of-tasks, but `ChampionARCDataset` expects individual files.

```python
# Load the big JSON file
with open(arc_data_dir / "arc-agi_training_challenges.json") as f:
    training_data = json.load(f)
# Result: {"task_id": {"train": [...], "test": [...]}, ...}

# Create temp files (one per task)
temp_dir = Path(__file__).parent.parent / "data" / "arc_agi_temp"
train_files = []
for task_id, task_data in training_data.items():
    task_file = temp_dir / f"{task_id}.json"
    with open(task_file, 'w') as f:
        json.dump(task_data, f)  # Write {"train": [...], "test": [...]}
    train_files.append(task_file)
```

**Why temp files?** Reuses existing `ChampionARCDataset` without rewriting it.

---

### **Step 2: Dataset Loading** (src/data/champion_data.py)

**CRITICAL FIX** (lines 79-86):

```python
# OLD (BROKEN for ARC-AGI-2):
query_examples = train_examples + test_examples  # KeyError on test!

# NEW (WORKS for both):
query_examples = []
for ex in train_examples + test_examples:
    if 'output' in ex:  # Only use examples with outputs
        query_examples.append(ex)
```

**Why this works:**
- **Re-ARC:** All examples have outputs → all used
- **ARC-AGI-2:** Test examples lack outputs → only train used

---

### **Step 3: Example Processing** (champion_data.py lines 123-200)

Each example is processed identically:

```python
def _process_example(query, context_examples, task_id):
    # 1. Extract grids
    input_grid = query['input']   # Always present
    output_grid = query['output']  # Only if 'output' in query (checked above)
    
    # 2. Flatten to 1D tensors
    src = torch.tensor([token for row in input_grid for token in row])
    tgt = torch.tensor([token for row in output_grid for token in row])
    
    # 3. Process context pairs (first num_context_pairs from train)
    ctx_inputs = []
    ctx_outputs = []
    for ctx_ex in context_examples[:2]:  # 2 pairs (Trial 69)
        ctx_in_padded = self._pad_grid(ctx_ex['input'], max_h, max_w)
        ctx_out_padded = self._pad_grid(ctx_ex['output'], max_h, max_w)
        ctx_inputs.append(ctx_in_padded)
        ctx_outputs.append(ctx_out_padded)
    
    # 4. Stack context: (2, H, W)
    ctx_input = torch.stack(ctx_inputs)
    ctx_output = torch.stack(ctx_outputs)
    
    return (src, tgt, ctx_input, ctx_output, src_shape, tgt_shape, task_id)
```

**NO DIFFERENCE in processing logic** - only which examples are selected differs.

---

## 3. Key Differences Summary

| Aspect | Re-ARC | ARC-AGI-2 |
|--------|--------|-----------|
| **File Format** | 400 individual JSON files | 2 big JSON dicts |
| **Tasks** | 400 synthetic | 1000 train, 120 eval |
| **Examples per Task** | 150 (all in train) | 2-5 (in train) + 1 (in test) |
| **Test Examples** | Empty or have outputs | Have inputs only (no outputs) |
| **Processing** | Load files directly | Convert dict → temp files |
| **Query Examples** | All train examples | Only train (test lacks outputs) |
| **Context Pairs** | 2 (from train) | 2 (from train) |
| **Grid Size** | 30x30 max | 30x30 max |

---

## 4. Verification Tests

### **Test 1: ARC-AGI-2 Loading** (test_arc_agi_loading.py)
```bash
python test_arc_agi_loading.py
```

**Results:**
- ✅ Loaded 1000 training tasks, 120 eval tasks
- ✅ Correctly identified test examples lack outputs
- ✅ Created temp files and loaded dataset
- ✅ Got 21 examples from 5 tasks (only train examples used)
- ✅ Sample shapes match expected format

### **Test 2: Re-ARC Regression** (test_rearc_loading.py)
```bash
python test_rearc_loading.py
```

**Results:**
- ✅ Loaded 308 train, 92 val tasks
- ✅ Got 750 examples from 5 tasks (all train examples used)
- ✅ Sample shapes match expected format
- ✅ No regression - re-arc still works correctly

---

## 5. Data Flow Diagram

```
ARC-AGI-2 Training Flow:
┌─────────────────────────────────────────────────────────────┐
│ arc-agi_training_challenges.json (1000 tasks)               │
│ {"task_1": {"train": [...], "test": [...]}, ...}           │
└────────────────────┬────────────────────────────────────────┘
                     │ Convert to individual files
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ data/arc_agi_temp/                                          │
│   task_1.json: {"train": [...], "test": [...]}            │
│   task_2.json: {"train": [...], "test": [...]}            │
│   ...                                                       │
└────────────────────┬────────────────────────────────────────┘
                     │ Load with ChampionARCDataset
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ For each task:                                              │
│   context = first 2 train examples                          │
│   queries = train examples ONLY (test lacks outputs)        │
│                                                             │
│ For each query:                                            │
│   src, tgt = flatten query input/output                    │
│   ctx_in, ctx_out = pad & stack context pairs              │
│   yield (src, tgt, ctx_in, ctx_out, shapes, task_id)      │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Critical Design Decisions

### **Decision 1: Why convert to temp files?**
**Rationale:** Reuse existing `ChampionARCDataset` without major refactoring.

**Alternative:** Rewrite dataset class to handle dict format directly.

**Trade-off:** Temp files add I/O overhead but maintain code simplicity.

### **Decision 2: Why only use train examples as queries?**
**Rationale:** ARC-AGI-2 test examples lack outputs (for prediction only).

**Alternative:** Could use test examples for evaluation later, but not training.

**Trade-off:** Fewer examples per task (~3 train vs ~150 re-arc), but this is the real ARC format.

### **Decision 3: Why 2 context pairs?**
**Rationale:** Match Trial 69 training (champion_bootstrap used 2 fixed pairs).

**Alternative:** Use all available train examples as context.

**Trade-off:** More context might help, but breaks consistency with champion baseline.

---

## 7. Example Counts

### **Re-ARC (308 train tasks):**
```
Total examples = 308 tasks × 150 examples/task = 46,200 examples
Context: First 2 of 150 train examples
Queries: All 150 train examples (148 after removing context)
```

### **ARC-AGI-2 (1000 train tasks):**
```
Total examples = 1000 tasks × ~3 train examples/task ≈ 3,000 examples
Context: First 2 of ~3-5 train examples
Queries: Remaining ~1-3 train examples (test excluded)
```

**Key difference:** ~15x fewer examples per task in real ARC!

---

## 8. Potential Issues & Solutions

### **Issue 1: Too Few Examples**
**Symptom:** ARC-AGI-2 has only ~1-3 query examples per task (after context).

**Impact:** May be insufficient for learning without regularization.

**Solution:** This is why we do transfer learning from champion_bootstrap (pre-trained on 46k re-arc examples).

### **Issue 2: Temp Files Overhead**
**Symptom:** Creating 1000 temp files adds ~5 seconds startup time.

**Impact:** Minor delay at training start.

**Solution:** Could cache temp files or refactor dataset class (not critical now).

### **Issue 3: Different Example Density**
**Symptom:** Re-ARC has 150 examples/task, ARC-AGI-2 has 3 examples/task.

**Impact:** Different data distribution between pre-training and transfer.

**Solution:** This is expected - testing transfer from synthetic to real data.

---

## 9. Files Modified/Created

### **Modified:**
1. `scripts/train_exp3_champion.py` (lines 91-127)
   - Added ARC-AGI-2 loading logic
   - Fixed path to use relative location

2. `src/data/champion_data.py` (lines 79-86)
   - Added check for 'output' key before using examples
   - Fixes KeyError on test examples

### **Created:**
1. `data/arc_prize_2025/` - ARC-AGI-2 dataset (6.5 MB)
2. `test_arc_agi_loading.py` - Verification test
3. `test_rearc_loading.py` - Regression test
4. `docs/ARC_AGI_DATA_LOADING_EXPLAINED.md` (this file)

---

## 10. Validation Checklist

- [x] Data copied to reproduction folder
- [x] Path changed from absolute to relative
- [x] Test examples without outputs handled correctly
- [x] Re-arc loading still works (regression test)
- [x] ARC-AGI-2 loading works (new test)
- [x] Context pairs = 2 (matching Trial 69)
- [x] Max grid size = 30 (matching Trial 69)
- [x] Example counts verified (3k vs 46k)
- [x] Temp file creation works
- [x] Both train and eval datasets load

---

**Status:** ✅ Ready for transfer learning experiments
