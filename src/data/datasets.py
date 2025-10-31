# jarc_reactor/data/datasets.py
from __future__ import annotations

from typing import List
import torch
from torch.utils.data import Dataset


class DynamicContextDataset(Dataset):
    """Dataset that supports a variable number of context pairs per sample.

    Each item returns:
      - src: Tensor[H, W] (long)
      - tgt: Tensor[H, W] (long)
      - ctx_input: Tensor[C_i, H, W] (long), number of context pairs C_i varies per sample
      - ctx_output: Tensor[C_i, H, W] (long)
      - task_id: Tensor[] (long scalar)
    """

    def __init__(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        ctx_inputs_list: List[torch.Tensor],
        ctx_outputs_list: List[torch.Tensor],
        task_ids: torch.Tensor,
    ) -> None:
        assert isinstance(inputs, torch.Tensor) and inputs.dim() == 3, "inputs must be [N, H, W]"
        assert isinstance(targets, torch.Tensor) and targets.dim() == 3, "targets must be [N, H, W]"
        assert isinstance(task_ids, torch.Tensor) and task_ids.dim() == 1, "task_ids must be [N]"
        n = inputs.size(0)
        assert targets.size(0) == n and task_ids.size(0) == n, "Mismatched dataset lengths"
        assert len(ctx_inputs_list) == n and len(ctx_outputs_list) == n, "Mismatched context lists length"

        self.inputs = inputs.long()
        self.targets = targets.long()
        self.ctx_inputs_list = [t.long() for t in ctx_inputs_list]
        self.ctx_outputs_list = [t.long() for t in ctx_outputs_list]
        self.task_ids = task_ids.long()

        # Precompute per-sample context lengths for efficient bucketing/sampling
        self.ctx_lengths: List[int] = [int(t.shape[0]) for t in self.ctx_inputs_list]

        # Sanity checks on shapes
        H, W = self.inputs.size(1), self.inputs.size(2)
        for i, (ci, co) in enumerate(zip(self.ctx_inputs_list, self.ctx_outputs_list)):
            assert ci.dim() == 3 and co.dim() == 3, f"Context tensors must be [C, H, W] at index {i}"
            assert ci.size(1) == H and ci.size(2) == W and co.size(1) == H and co.size(2) == W, (
                f"Context grid size mismatch at index {i}: expected ({H},{W}), got "
                f"ctx_in=({ci.size(1)},{ci.size(2)}), ctx_out=({co.size(1)},{co.size(2)})"
            )

    def __len__(self) -> int:
        return self.inputs.size(0)

    def __getitem__(self, idx: int):
        return (
            self.inputs[idx],
            self.targets[idx],
            self.ctx_inputs_list[idx],
            self.ctx_outputs_list[idx],
            self.task_ids[idx],
        )
