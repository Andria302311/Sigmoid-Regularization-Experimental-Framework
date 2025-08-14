from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .base import BasePruner


class RandomPruner(BasePruner):
	def __init__(self, target_sparsity: float, seed: int | None = None) -> None:
		super().__init__(target_sparsity=target_sparsity, pruning_threshold=None)
		self.generator = torch.Generator()
		if seed is not None:
			self.generator.manual_seed(seed)

	def compute_sparsity_loss(self, model: nn.Module, epoch: int) -> torch.Tensor:
		device = next(model.parameters()).device
		return torch.zeros(1, device=device)

	def create_pruning_masks(self, model: nn.Module) -> Dict[str, torch.Tensor]:
		masks: Dict[str, torch.Tensor] = {}
		for name, param in self.iter_prunable_params(model):
			numel = param.numel()
			k = int(self.target_sparsity * numel)
			keep = torch.ones(numel, dtype=torch.float32, device=param.device)
			if k > 0:
				drop_idx = torch.randperm(numel, generator=self.generator, device=param.device)[:k]
				keep[drop_idx] = 0.0
			masks[name] = keep.view_as(param)
		self.masks = masks
		return masks