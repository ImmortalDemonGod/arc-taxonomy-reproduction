"""
Minimal Decoder-Only Data Loader

Following TDD and cs336 style: simplest implementation that satisfies contracts.

Contract Requirements (from test_data_loaders.py):
1. Sequence format: [INPUT_GRID] [SEP] [OUTPUT_GRID]
2. Batch shape: (batch_size, seq_len)
3. dtype: torch.long
"""
import json
import torch
from pathlib import Path
from typing import List, Tuple, Dict, Any
from torch.utils.data import Dataset


class DecoderOnlyARCDataset(Dataset):
    """
    Minimal dataset for Decoder-Only baseline.
    
    Loads ARC tasks and formats as: [input_flatten] [SEP] [output_flatten]
    
    Following cs336 style:
    - Clear docstrings
    - Type hints
    - Minimal but complete
    """
    
    def __init__(
        self,
        task_files: List[Path],
        sep_token: int = 10,
        pad_token: int = 10,
        max_seq_len: int = 512,
    ):
        """
        Initialize dataset.
        
        Args:
            task_files: List of paths to ARC JSON files
            sep_token: Separator token ID
            pad_token: Padding token ID  
            max_seq_len: Maximum sequence length
        """
        self.task_files = task_files
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.max_seq_len = max_seq_len
        
        # Load and linearize all examples
        self.sequences: List[torch.Tensor] = []
        self._load_data()
    
    def _load_data(self):
        """Load and linearize all ARC examples."""
        for task_file in self.task_files:
            try:
                with open(task_file) as f:
                    task_data = json.load(f)
                
                # Process training examples
                for example in task_data.get('train', []):
                    seq = self._linearize_example(
                        example['input'],
                        example['output']
                    )
                    if seq.size(0) <= self.max_seq_len:
                        self.sequences.append(seq)
                
            except Exception as e:
                print(f"Warning: Failed to load {task_file}: {e}")
                continue
    
    def _linearize_example(
        self, 
        input_grid: List[List[int]], 
        output_grid: List[List[int]]
    ) -> torch.Tensor:
        """
        Linearize input/output pair into sequence.
        
        Format: [input_flatten] [SEP] [output_flatten]
        
        Args:
            input_grid: 2D list of integers
            output_grid: 2D list of integers
            
        Returns:
            1D tensor of token IDs
        """
        # Flatten grids row-wise
        input_flat = [token for row in input_grid for token in row]
        output_flat = [token for row in output_grid for token in row]
        
        # Concatenate with SEP
        sequence = input_flat + [self.sep_token] + output_flat
        
        return torch.tensor(sequence, dtype=torch.long)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get sequence by index.
        
        Returns:
            Tensor of shape (seq_len,) with token IDs
        """
        return self.sequences[idx]


def collate_decoder_only(batch: List[torch.Tensor], pad_token: int = 10) -> torch.Tensor:
    """
    Collate function for decoder-only batches.
    
    Pads sequences to max length in batch.
    
    Args:
        batch: List of 1D tensors (varying lengths)
        pad_token: Token ID to use for padding
        
    Returns:
        Tensor of shape (batch_size, max_len)
    """
    # Find max length in batch
    max_len = max(seq.size(0) for seq in batch)
    
    # Pad all sequences
    padded_batch = []
    for seq in batch:
        if seq.size(0) < max_len:
            padding = torch.full((max_len - seq.size(0),), pad_token, dtype=torch.long)
            padded_seq = torch.cat([seq, padding])
        else:
            padded_seq = seq
        padded_batch.append(padded_seq)
    
    return torch.stack(padded_batch)


def create_decoder_only_dataloader(
    task_files: List[Path],
    batch_size: int = 8,
    shuffle: bool = True,
    **dataset_kwargs
) -> torch.utils.data.DataLoader:
    """
    Factory function for creating decoder-only dataloader.
    
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
    
    dataset = DecoderOnlyARCDataset(task_files, **dataset_kwargs)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_decoder_only(batch, dataset.pad_token),
    )
