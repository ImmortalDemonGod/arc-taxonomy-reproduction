"""
Dataset for training TaskEncoder on ARC tasks.

Loads tasks from JSON files and returns demonstration pairs for classification.
"""
import json
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class ARCTaskDataset(Dataset):
    """Dataset for training TaskEncoder on ARC tasks."""
    
    def __init__(self, task_files: List[Path], categories_json: Path, max_grid_size: int = 30):
        """
        Args:
            task_files: List of paths to task JSON files
            categories_json: Path to task_categories_v4.json
            max_grid_size: Maximum grid dimension (pad to this size)
        """
        self.task_files = task_files
        self.max_grid_size = max_grid_size
        
        # Load category labels
        with open(categories_json) as f:
            task_categories = json.load(f)
        
        # Map category names to indices
        self.category_to_idx = {
            'S1': 0, 'S2': 1, 'S3': 2,
            'C1': 3, 'C2': 4,
            'K1': 5,
            'L1': 6,
            'A1': 7, 'A2': 8
        }
        
        # Build dataset: (task_file, category_idx)
        self.examples = []
        for task_file in task_files:
            task_id = task_file.stem
            if task_id in task_categories:
                category = task_categories[task_id]
                category_idx = self.category_to_idx[category]
                self.examples.append((task_file, category_idx))
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            demo_input: (num_demos, max_grid_size, max_grid_size) - padded input grids
            demo_output: (num_demos, max_grid_size, max_grid_size) - padded output grids
            category_idx: int - category label
        """
        task_file, category_idx = self.examples[idx]
        
        # Load task
        with open(task_file) as f:
            task = json.load(f)
        
        # Use first 3 demonstrations (standard for ARC)
        demonstrations = task['train'][:3]
        
        # Pad grids
        demo_inputs = []
        demo_outputs = []
        
        for demo in demonstrations:
            input_grid = torch.tensor(demo['input'], dtype=torch.long)
            output_grid = torch.tensor(demo['output'], dtype=torch.long)
            
            demo_inputs.append(self._pad_grid(input_grid))
            demo_outputs.append(self._pad_grid(output_grid))
        
        demo_input = torch.stack(demo_inputs)  # (num_demos, H, W)
        demo_output = torch.stack(demo_outputs)
        
        return demo_input, demo_output, category_idx
    
    def _pad_grid(self, grid: torch.Tensor) -> torch.Tensor:
        """Pad grid to max_grid_size × max_grid_size."""
        h, w = grid.shape
        if h > self.max_grid_size or w > self.max_grid_size:
            # Truncate if too large (rare)
            grid = grid[:self.max_grid_size, :self.max_grid_size]
            h, w = grid.shape
        
        # Pad with 10 (padding token)
        pad_h = self.max_grid_size - h
        pad_w = self.max_grid_size - w
        padded = F.pad(grid, (0, pad_w, 0, pad_h), value=10)
        
        return padded


def collate_arc_tasks(batch):
    """Collate function for ARCTaskDataset."""
    demo_inputs, demo_outputs, category_indices = zip(*batch)
    
    demo_inputs = torch.stack(demo_inputs)  # (batch, num_demos, H, W)
    demo_outputs = torch.stack(demo_outputs)
    category_indices = torch.tensor(category_indices, dtype=torch.long)
    
    return demo_inputs, demo_outputs, category_indices
