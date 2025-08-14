from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .base import BasePruner


class MagnitudePruner(BasePruner):
    def __init__(self, target_sparsity: float, global_pruning: bool = True) -> None:
        super().__init__(target_sparsity=target_sparsity, pruning_threshold=None)
        self.global_pruning = bool(global_pruning)

    def compute_sparsity_loss(self, model: nn.Module, epoch: int) -> torch.Tensor:
        device = next(model.parameters()).device
        return torch.zeros(1, device=device)

    def create_pruning_masks(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        masks: Dict[str, torch.Tensor] = {}
        params = list(self.iter_prunable_params(model))
        if not params:
            return masks
        if self.global_pruning:
            all_weights = self._gather_all_weights(model)
            if all_weights.numel() == 0:
                threshold = 0.0
            else:
                k = int(self.target_sparsity * all_weights.numel())
                if k <= 0:
                    threshold = -float("inf")
                elif k >= all_weights.numel():
                    threshold = float("inf")
                else:
                    threshold = torch.kthvalue(all_weights, k).values.item()
            for name, param in params:
                mask = (param.detach().abs() > threshold).to(param.device)
                masks[name] = mask.to(param.dtype)
        else:
            for name, param in params:
                flat = param.detach().abs().flatten()
                k = int(self.target_sparsity * flat.numel())
                if k <= 0:
                    threshold = -float("inf")
                elif k >= flat.numel():
                    threshold = float("inf")
                else:
                    threshold = torch.kthvalue(flat, k).values.item()
                mask = (param.detach().abs() > threshold).to(param.device)
                masks[name] = mask.to(param.dtype)
        self.masks = masks
        return masks