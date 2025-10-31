# JARC-Reactor/Utils/context_data.py
from dataclasses import dataclass, field
from typing import Optional, Tuple
import torch
import logging

logger = logging.getLogger(__name__)

@dataclass
class ContextPair:
    context_input: Optional[torch.Tensor] = None
    context_output: Optional[torch.Tensor] = None
    # Expected shape no longer fixed; we require a 3-D tensor [N, H, W] but N can vary
    expected_hw: Tuple[int, int] = field(default_factory=lambda: (30, 30))

    def _determine_default_tensor_attributes(self) -> Tuple[torch.dtype, str]:
        """Determines default dtype and device based on existing tensors."""
        default_dtype = torch.float32
        default_device = "cpu"

        if self.context_input is not None and isinstance(self.context_input, torch.Tensor):
            return self.context_input.dtype, str(self.context_input.device)
        if self.context_output is not None and isinstance(self.context_output, torch.Tensor):
            return self.context_output.dtype, str(self.context_output.device)
        
        return default_dtype, default_device

    def _validate_or_create_tensor(
        self,
        tensor: Optional[torch.Tensor],
        tensor_name: str,
        default_dtype: torch.dtype,
        default_device: str
    ) -> torch.Tensor:
        """Validates an existing tensor or creates a default one."""
        if tensor is not None:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{tensor_name} must be a torch.Tensor or None, got {type(tensor)}")
            if len(tensor.shape) != 3 or tensor.shape[1:] != self.expected_hw:
                raise ValueError(
                    f"{tensor_name} must have shape [N, H, W] with H,W={self.expected_hw}, got {tensor.shape}"
                )
            return tensor
        else:
            logger.warning(
                f"{tensor_name} is None. Using default tensor of shape "
                f"[2, {self.expected_hw[0]}, {self.expected_hw[1]}], dtype {default_dtype}, device {default_device}."
            )
            # create empty [N,H,W] default (variable number of context pairs)
        default_shape = (2, *self.expected_hw)
        return torch.zeros(
                default_shape,
                dtype=default_dtype,
                device=default_device
            )

    def __post_init__(self):
        default_dtype, default_device = self._determine_default_tensor_attributes()

        self.context_input = self._validate_or_create_tensor(
            self.context_input, "context_input", default_dtype, default_device
        )
        self.context_output = self._validate_or_create_tensor(
            self.context_output, "context_output", default_dtype, default_device
        )

