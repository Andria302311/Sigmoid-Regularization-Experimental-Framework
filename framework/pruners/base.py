from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn


class BasePruner(ABC):
    """Abstract base pruner providing common utilities for structural pruning.

    Subclasses must implement the sparsity-inducing loss (if any) and mask creation logic.
    """

    def __init__(self, target_sparsity: float | None = None, pruning_threshold: float | None = None) -> None:
        self.target_sparsity = target_sparsity
        self.pruning_threshold = pruning_threshold
        self.masks: Dict[str, torch.Tensor] = {}

    @abstractmethod
    def compute_sparsity_loss(self, model: nn.Module, epoch: int) -> torch.Tensor:
        """Compute an auxiliary loss encouraging sparsity. Return a scalar tensor on the same device as the model.
        For pruners that do not use a sparsity loss, return torch.zeros(1, device=model.device) equivalently.
        """

    @abstractmethod
    def create_pruning_masks(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Create binary masks for parameters to prune. Keys are parameter full names.
        """

    def apply_masks(self, model: nn.Module) -> None:
        """Apply precomputed masks in-place to model parameters."""
        if not self.masks:
            self.masks = self.create_pruning_masks(model)
        named_params = dict(model.named_parameters())
        for name, mask in self.masks.items():
            if name in named_params:
                param = named_params[name]
                param.data.mul_(mask.to(param.data.device))

    def get_sparsity_stats(self, model: nn.Module) -> Dict[str, float]:
        total_params = 0
        zero_params = 0
        for _, param in self.iter_prunable_params(model):
            numel = param.numel()
            total_params += numel
            zero_params += int((param == 0).sum().item())
        sparsity = (zero_params / total_params) if total_params > 0 else 0.0
        return {
            "total_params": float(total_params),
            "zero_params": float(zero_params),
            "sparsity_level": float(sparsity),
        }

    @staticmethod
    def iter_prunable_params(model: nn.Module) -> Iterable[Tuple[str, torch.Tensor]]:
        """Iterate over parameters eligible for pruning.

        Default policy: any parameter named like a weight tensor (contains "weight") and has at least 2 dims.
        This covers Linear/Conv layers and RNN weights. Subclasses or models may override by exposing
        an attribute `prunable_parameter_names` (set of fully qualified param names).
        """
        prunable_names = getattr(model, "prunable_parameter_names", None)
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if prunable_names is not None and name not in prunable_names:
                continue
            if param.dim() >= 2 and ("weight" in name or prunable_names is not None):
                yield name, param

    @staticmethod
    def _gather_all_weights(model: nn.Module) -> torch.Tensor:
        weights = []
        for _, param in BasePruner.iter_prunable_params(model):
            weights.append(param.detach().abs().flatten())
        if not weights:
            return torch.tensor([], device=next(model.parameters()).device)
        return torch.cat(weights)