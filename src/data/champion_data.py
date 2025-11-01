"""
Minimal Champion Data Loader (with Context Pairs)

Following TDD and cs336 style: simplest implementation that satisfies contracts.

Contract Requirements (from test_data_loaders.py):
1. Separate src/tgt format + context pairs
2. Context format: (batch, num_pairs, H, W)
3. CRITICAL: context input/output must have matching H, W
4. dtype: torch.long
"""
import json
import torch
from pathlib import Path
from typing import List, Tuple, Dict, Any
from torch.utils.data import Dataset


class ChampionARCDataset(Dataset):
    """
    Minimal dataset for Champion model with context pairs.
    
    Returns (src, tgt, ctx_input, ctx_output) where context pairs are grids.
    
    Following cs336 style:
    - Clear docstrings
    - Type hints
    - Minimal but complete
    """
    
    def __init__(
        self,
        task_files: List[Path],
        num_context_pairs: int = 2,  # Champion uses 2
        max_grid_size: int = 30,
        pad_token: int = 10,
        task_categories: Dict[str, str] = None,  # Optional: task_id -> category mapping
    ):
        """
        Initialize dataset.
        
        Args:
            task_files: List of paths to ARC JSON files
            num_context_pairs: Number of context pairs to use
            max_grid_size: Maximum grid dimension (H, W)
            pad_token: Padding token ID
            task_categories: Optional task_id -> category mapping
        """
        self.task_files = task_files
        self.num_context_pairs = num_context_pairs
        self.max_grid_size = max_grid_size
        self.pad_token = pad_token
        self.task_categories = task_categories or {}
        
        # Load all examples: (src, tgt, ctx_input, ctx_output, src_shape, tgt_shape, task_id)
        self.examples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple, tuple, str]] = []
        self._load_data()
    
    def _load_data(self):
        """Load all ARC tasks with context pairs."""
        for task_file in self.task_files:
            try:
                # Extract task_id from filename
                task_id = task_file.stem  # e.g., "007bbfb7" from "007bbfb7.json"
                
                with open(task_file) as f:
                    task_data = json.load(f)
                
                train_examples = task_data.get('train', [])
                test_examples = task_data.get('test', [])
                
                # Need at least num_context_pairs examples
                if len(train_examples) < self.num_context_pairs:
                    continue
                
                # Use first num_context_pairs as context (demonstrations)
                context_examples = train_examples[:self.num_context_pairs]
                
                # Use training examples as queries
                # Note: For re-arc, test examples also have outputs so we can use them
                # For real ARC-AGI-2, test examples don't have outputs (they're for prediction)
                # So we only use examples that have 'output' key
                query_examples = []
                for ex in train_examples + test_examples:
                    if 'output' in ex:
                        query_examples.append(ex)
                
                for query in query_examples:
                    example = self._process_example(query, context_examples, task_id)
                    if example is not None:
                        self.examples.append(example)
                
            except Exception as e:
                print(f"Warning: Failed to load {task_file}: {e}")
                continue
    
    def _pad_grid(
        self, 
        grid: List[List[int]],
        target_h: int,
        target_w: int
    ) -> torch.Tensor:
        """
        Pad grid to target size.
        
        Args:
            grid: 2D list of integers
            target_h: Target height
            target_w: Target width
            
        Returns:
            Padded grid tensor of shape (target_h, target_w)
        """
        h, w = len(grid), len(grid[0]) if grid else 0
        
        # Create padded tensor
        padded = torch.full((target_h, target_w), self.pad_token, dtype=torch.long)
        
        # Copy original grid
        for i, row in enumerate(grid):
            for j, val in enumerate(row):
                if i < target_h and j < target_w:
                    padded[i, j] = val
        
        return padded
    
    def _process_example(
        self,
        query: Dict,
        context_examples: List[Dict],
        task_id: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple, tuple, str]:
        """
        Process query example with context.
        
        Args:
            query: Query input/output pair
            context_examples: List of context input/output pairs
            
        Returns:
            Tuple of (src, tgt, ctx_input, ctx_output) or None if invalid
        """
        try:
            # Get original grid shapes before flattening
            input_grid = query['input']
            output_grid = query['output']
            
            src_h = len(input_grid)
            src_w = len(input_grid[0]) if input_grid else 0
            tgt_h = len(output_grid)
            tgt_w = len(output_grid[0]) if output_grid else 0
            
            # Flatten query src/tgt
            src = torch.tensor(
                [token for row in input_grid for token in row],
                dtype=torch.long
            )
            tgt = torch.tensor(
                [token for row in output_grid for token in row],
                dtype=torch.long
            )
            
            # Store shapes
            src_shape = (src_h, src_w)
            tgt_shape = (tgt_h, tgt_w)
            
            # Process context pairs
            # CRITICAL: All pairs must have same H, W to stack
            # First pass: find max dimensions across ALL context pairs
            max_h, max_w = 1, 1
            for ctx_ex in context_examples[:self.num_context_pairs]:
                in_h = len(ctx_ex['input'])
                in_w = len(ctx_ex['input'][0]) if ctx_ex['input'] else 0
                out_h = len(ctx_ex['output'])
                out_w = len(ctx_ex['output'][0]) if ctx_ex['output'] else 0
                
                max_h = max(max_h, in_h, out_h)
                max_w = max(max_w, in_w, out_w)
            
            # Cap at max_grid_size
            max_h = min(max_h, self.max_grid_size)
            max_w = min(max_w, self.max_grid_size)
            
            # Second pass: pad all to same size
            ctx_inputs = []
            ctx_outputs = []
            
            for ctx_ex in context_examples[:self.num_context_pairs]:
                ctx_in_padded = self._pad_grid(ctx_ex['input'], max_h, max_w)
                ctx_out_padded = self._pad_grid(ctx_ex['output'], max_h, max_w)
                
                ctx_inputs.append(ctx_in_padded)
                ctx_outputs.append(ctx_out_padded)
            
            # Stack context pairs: (num_pairs, H, W)
            ctx_input = torch.stack(ctx_inputs)
            ctx_output = torch.stack(ctx_outputs)
            
            # Return with grid shapes and task_id
            return (src, tgt, ctx_input, ctx_output, src_shape, tgt_shape, task_id)
            
        except Exception as e:
            print(f"Warning: Failed to process example: {e}")
            return None
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int):
        """
        Get example by index.
        
        Returns:
            Tuple of (src, tgt, ctx_input, ctx_output)
        """
        return self.examples[idx]


def collate_champion(
    batch,
    pad_token: int = 10
):
    """
    Collate function for champion batches.
    
    Args:
        batch: List of (src, tgt, ctx_input, ctx_output, src_shape, tgt_shape, task_id) tuples
        pad_token: Token ID for padding
        
    Returns:
        Tuple of (src_batch, tgt_batch, ctx_input_batch, ctx_output_batch, src_shapes, tgt_shapes, task_ids)
    """
    srcs, tgts, ctx_inputs, ctx_outputs, src_shapes, tgt_shapes, task_ids = zip(*batch)
    
    # Pad src sequences
    max_src_len = max(src.size(0) for src in srcs)
    padded_srcs = []
    for src in srcs:
        if src.size(0) < max_src_len:
            padding = torch.full((max_src_len - src.size(0),), pad_token, dtype=torch.long)
            padded_src = torch.cat([src, padding])
        else:
            padded_src = src
        padded_srcs.append(padded_src)
    
    # Pad tgt sequences
    max_tgt_len = max(tgt.size(0) for tgt in tgts)
    padded_tgts = []
    for tgt in tgts:
        if tgt.size(0) < max_tgt_len:
            padding = torch.full((max_tgt_len - tgt.size(0),), pad_token, dtype=torch.long)
            padded_tgt = torch.cat([tgt, padding])
        else:
            padded_tgt = tgt
        padded_tgts.append(padded_tgt)
    
    # Pad context grids to max H, W in batch
    # Find max dimensions across all context examples
    max_ctx_h = max(ctx.size(1) for ctx in ctx_inputs)
    max_ctx_w = max(ctx.size(2) for ctx in ctx_inputs)
    
    padded_ctx_inputs = []
    padded_ctx_outputs = []
    
    for ctx_in, ctx_out in zip(ctx_inputs, ctx_outputs):
        num_pairs = ctx_in.size(0)
        curr_h, curr_w = ctx_in.size(1), ctx_in.size(2)
        
        if curr_h < max_ctx_h or curr_w < max_ctx_w:
            # Need to pad
            padded_in = torch.full((num_pairs, max_ctx_h, max_ctx_w), pad_token, dtype=torch.long)
            padded_out = torch.full((num_pairs, max_ctx_h, max_ctx_w), pad_token, dtype=torch.long)
            
            # Copy original data
            padded_in[:, :curr_h, :curr_w] = ctx_in
            padded_out[:, :curr_h, :curr_w] = ctx_out
            
            padded_ctx_inputs.append(padded_in)
            padded_ctx_outputs.append(padded_out)
        else:
            padded_ctx_inputs.append(ctx_in)
            padded_ctx_outputs.append(ctx_out)
    
    return (
        torch.stack(padded_srcs),
        torch.stack(padded_tgts),
        torch.stack(padded_ctx_inputs),
        torch.stack(padded_ctx_outputs),
        list(src_shapes),  # Keep as list of tuples
        list(tgt_shapes),  # Keep as list of tuples
        list(task_ids),    # Keep as list of strings
    )


def create_champion_dataloader(
    task_files: List[Path],
    batch_size: int = 8,
    shuffle: bool = True,
    task_categories: Dict[str, str] = None,  # Optional: task_id -> category mapping
    num_workers: int = 0,  # Parallel data loading
    pin_memory: bool = True,  # Faster GPU transfer
    **dataset_kwargs
) -> torch.utils.data.DataLoader:
    """
    Factory function for creating champion dataloader.
    
    Following cs336 style with clean interface.
    
    Args:
        task_files: List of ARC JSON file paths
        batch_size: Batch size
        shuffle: Whether to shuffle
        task_categories: Optional dict mapping task_id -> category
        **dataset_kwargs: Additional args for dataset
        
    Returns:
        DataLoader ready for training
        
    Note:
        ARC tasks have "train" and "test" keys in JSON, but these are NOT
        ML train/val splits! They're all demonstrations of the task's pattern.
        The actual train/val split happens at the TASK level (which tasks you load).
    """
    from torch.utils.data import DataLoader
    
    # Auto-load task_categories if not provided
    if task_categories is None and len(task_files) > 0:
        # Try to find task_categories.json in the same directory
        data_dir = task_files[0].parent
        categories_file = data_dir / "task_categories.json"
        if categories_file.exists():
            with open(categories_file) as f:
                task_categories = json.load(f)
    
    dataset = ChampionARCDataset(task_files, task_categories=task_categories, **dataset_kwargs)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_champion(batch, dataset.pad_token),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
