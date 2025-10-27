"""
Minimal Encoder-Decoder Data Loader

Following TDD and cs336 style: simplest implementation that satisfies contracts.

Contract Requirements (from test_data_loaders.py):
1. Separate src/tgt format (no linearization)
2. Batch shapes: (batch_size, src_len), (batch_size, tgt_len)
3. dtype: torch.long
"""
import json
import torch
from pathlib import Path
from typing import List, Tuple, Dict, Any
from torch.utils.data import Dataset


class EncoderDecoderARCDataset(Dataset):
    """
    Minimal dataset for Encoder-Decoder models.
    
    Returns separate src (input) and tgt (output) grids (flattened).
    
    Following cs336 style:
    - Clear docstrings
    - Type hints
    - Minimal but complete
    """
    
    def __init__(
        self,
        task_files: List[Path],
        pad_token: int = 10,
        max_seq_len: int = 900,  # 30x30 = 900 max
    ):
        """
        Initialize dataset.
        
        Args:
            task_files: List of paths to ARC JSON files
            pad_token: Padding token ID
            max_seq_len: Maximum sequence length for src or tgt
        """
        self.task_files = task_files
        self.pad_token = pad_token
        self.max_seq_len = max_seq_len
        
        # Load all examples as (src, tgt) pairs
        self.examples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._load_data()
    
    def _load_data(self):
        """Load all ARC examples as src/tgt pairs."""
        for task_file in self.task_files:
            try:
                with open(task_file) as f:
                    task_data = json.load(f)
                
                # Process training examples
                for example in task_data.get('train', []):
                    src, tgt = self._process_example(
                        example['input'],
                        example['output']
                    )
                    # Only include if within max length
                    if src.size(0) <= self.max_seq_len and tgt.size(0) <= self.max_seq_len:
                        self.examples.append((src, tgt))
                
            except Exception as e:
                print(f"Warning: Failed to load {task_file}: {e}")
                continue
    
    def _process_example(
        self,
        input_grid: List[List[int]],
        output_grid: List[List[int]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process input/output pair into src/tgt tensors.
        
        Args:
            input_grid: 2D list of integers
            output_grid: 2D list of integers
            
        Returns:
            Tuple of (src, tgt) where both are 1D tensors
        """
        # Flatten grids row-wise
        src = [token for row in input_grid for token in row]
        tgt = [token for row in output_grid for token in row]
        
        return (
            torch.tensor(src, dtype=torch.long),
            torch.tensor(tgt, dtype=torch.long)
        )
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get (src, tgt) pair by index.
        
        Returns:
            Tuple of (src, tgt) where:
            - src: 1D tensor of input tokens
            - tgt: 1D tensor of output tokens
        """
        return self.examples[idx]


def collate_encoder_decoder(
    batch: List[Tuple[torch.Tensor, torch.Tensor]], 
    pad_token: int = 10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for encoder-decoder batches.
    
    Pads src and tgt sequences separately.
    
    Args:
        batch: List of (src, tgt) tuples
        pad_token: Token ID to use for padding
        
    Returns:
        Tuple of (src_batch, tgt_batch) where:
        - src_batch: (batch_size, max_src_len)
        - tgt_batch: (batch_size, max_tgt_len)
    """
    srcs, tgts = zip(*batch)
    
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
    
    return torch.stack(padded_srcs), torch.stack(padded_tgts)


def create_encoder_decoder_dataloader(
    task_files: List[Path],
    batch_size: int = 8,
    shuffle: bool = True,
    **dataset_kwargs
) -> torch.utils.data.DataLoader:
    """
    Factory function for creating encoder-decoder dataloader.
    
    Following cs336 style with clean interface.
    
    Args:
        task_files: List of ARC JSON file paths
        batch_size: Batch size
        shuffle: Whether to shuffle
        **dataset_kwargs: Additional args for dataset
        
    Returns:
        DataLoader ready for training
    """
    from torch.utils.data import DataLoader
    
    dataset = EncoderDecoderARCDataset(task_files, **dataset_kwargs)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_encoder_decoder(batch, dataset.pad_token),
        num_workers=0,  # Use 0 to avoid multiprocessing pickle errors
        pin_memory=True,
    )
