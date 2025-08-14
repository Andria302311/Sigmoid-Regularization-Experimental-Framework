from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .base import BasePruner


class SigmoidRegularizationPruner(BasePruner):
    def __init__(
        self,
        epsilon: float,
        alpha: float,
        a: float,
        pruning_threshold: float | None,
        target_sparsity: float | None,
        warmup_epochs: int = 0,
    ) -> None:
        super().__init__(target_sparsity=target_sparsity, pruning_threshold=pruning_threshold)
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.a = float(a)
        self.warmup_epochs = int(warmup_epochs)

    def update_epsilon(self, epoch: int) -> float:
        return float(self.epsilon * (self.a ** epoch))

    def compute_sparsity_loss(self, model: nn.Module, epoch: int) -> torch.Tensor:
        device = next(model.parameters()).device
        if epoch < self.warmup_epochs:
            return torch.zeros(1, device=device)
        epsilon_t = self.update_epsilon(epoch)
        total = torch.tensor(0.0, device=device)
        count = 0
        for _, param in self.iter_prunable_params(model):
            count += param.numel()
            total = total + torch.sigmoid(param.abs() / max(epsilon_t, 1e-12)).sum()
        if count == 0:
            return torch.zeros(1, device=device)
        return self.alpha * (total / count)

    def create_pruning_masks(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        masks: Dict[str, torch.Tensor] = {}
        named_params = list(self.iter_prunable_params(model))
        if not named_params:
            return masks
        if self.target_sparsity is not None and 0.0 < self.target_sparsity < 1.0:
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
        else:
            threshold = float(self.pruning_threshold or 0.0)
        for name, param in named_params:
            mask = (param.detach().abs() > threshold).to(param.device)
            masks[name] = mask.to(param.dtype)
        self.masks = masks
        return masks