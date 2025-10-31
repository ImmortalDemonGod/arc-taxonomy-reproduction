"""
Single-task dataset for LoRA fine-tuning.

Follows champion_data.py pattern: returns (src, tgt, ctx_in, ctx_out, shapes, task_id).
Following cs336 style: clear, minimal, well-documented.
"""
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class SingleTaskDataset(Dataset):
    """Dataset for fine-tuning on one ARC task."""
    
    def __init__(self, task_file: Path, num_context_pairs: int = 2, pad_token: int = 10):
        """
        Args:
            task_file: Path to task JSON
            num_context_pairs: Number of demo pairs for context (default 2)
            pad_token: Padding token ID (default 10)
        """
        with open(task_file) as f:
            data = json.load(f)
        
        self.task_id = task_file.stem
        self.pad_token = pad_token
        train = data['train']
        
        if len(train) < num_context_pairs:
            self.examples = []
            return
        
        # Use first num_context_pairs as context
        context = train[:num_context_pairs]
        
        # Process all train examples
        self.examples = []
        for query in train:
            ex = self._process(query, context)
            if ex:
                self.examples.append(ex)
    
    def _process(self, query, context):
        """Convert to (src, tgt, ctx_in, ctx_out, src_shape, tgt_shape, task_id)."""
        try:
            # Flatten src/tgt to sequences
            src = torch.tensor(
                [v for row in query['input'] for v in row],
                dtype=torch.long
            )
            tgt = torch.tensor(
                [v for row in query['output'] for v in row],
                dtype=torch.long
            )
            
            # Store shapes
            src_shape = (len(query['input']), len(query['input'][0]))
            tgt_shape = (len(query['output']), len(query['output'][0]))
            
            # Pad context grids to max size
            max_h = max(max(len(c['input']), len(c['output'])) for c in context)
            max_w = max(max(len(c['input'][0]), len(c['output'][0])) for c in context)
            max_h, max_w = min(max_h, 30), min(max_w, 30)
            
            ctx_in = torch.stack([self._pad(c['input'], max_h, max_w) for c in context])
            ctx_out = torch.stack([self._pad(c['output'], max_h, max_w) for c in context])
            
            return (src, tgt, ctx_in, ctx_out, src_shape, tgt_shape, self.task_id)
        except:
            return None
    
    def _pad(self, grid, h, w):
        """Pad grid to h x w."""
        padded = torch.full((h, w), self.pad_token, dtype=torch.long)
        for i, row in enumerate(grid):
            for j, val in enumerate(row):
                if i < h and j < w:
                    padded[i, j] = val
        return padded
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def collate_single_task(batch, pad_token: int = 10):
    """Collate function. Based on champion_data.py collate_champion."""
    srcs, tgts, ctx_ins, ctx_outs, src_shapes, tgt_shapes, task_ids = zip(*batch)
    
    # Pad sequences
    max_src = max(s.size(0) for s in srcs)
    max_tgt = max(t.size(0) for t in tgts)
    
    padded_srcs = []
    for src in srcs:
        if src.size(0) < max_src:
            pad = torch.full((max_src - src.size(0),), pad_token, dtype=torch.long)
            padded_srcs.append(torch.cat([src, pad]))
        else:
            padded_srcs.append(src)
    
    padded_tgts = []
    for tgt in tgts:
        if tgt.size(0) < max_tgt:
            pad = torch.full((max_tgt - tgt.size(0),), pad_token, dtype=torch.long)
            padded_tgts.append(torch.cat([tgt, pad]))
        else:
            padded_tgts.append(tgt)
    
    # Pad context grids
    max_h = max(c.size(1) for c in ctx_ins)
    max_w = max(c.size(2) for c in ctx_ins)
    
    padded_ctx_ins = []
    padded_ctx_outs = []
    for ctx_in, ctx_out in zip(ctx_ins, ctx_outs):
        n, h, w = ctx_in.shape
        if h < max_h or w < max_w:
            pad_in = torch.full((n, max_h, max_w), pad_token, dtype=torch.long)
            pad_out = torch.full((n, max_h, max_w), pad_token, dtype=torch.long)
            pad_in[:, :h, :w] = ctx_in
            pad_out[:, :h, :w] = ctx_out
            padded_ctx_ins.append(pad_in)
            padded_ctx_outs.append(pad_out)
        else:
            padded_ctx_ins.append(ctx_in)
            padded_ctx_outs.append(ctx_out)
    
    return (
        torch.stack(padded_srcs),
        torch.stack(padded_tgts),
        torch.stack(padded_ctx_ins),
        torch.stack(padded_ctx_outs),
        src_shapes,
        tgt_shapes,
        task_ids
    )
