from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .base import BasePruner


class L1RegularizationPruner(BasePruner):
	def __init__(self, lambda_l1: float, target_sparsity: float | None = None, pruning_threshold: float | None = None) -> None:
		super().__init__(target_sparsity=target_sparsity, pruning_threshold=pruning_threshold)
		self.lambda_l1 = float(lambda_l1)

	def compute_sparsity_loss(self, model: nn.Module, epoch: int) -> torch.Tensor:
		total = 0.0
		device = next(model.parameters()).device
		count = 0
		for _, param in self.iter_prunable_params(model):
			count += 1
			total = total + param.abs().sum()
		if count == 0:
			return torch.zeros(1, device=device)
		return self.lambda_l1 * total

	def create_pruning_masks(self, model: nn.Module) -> Dict[str, torch.Tensor]:
		masks: Dict[str, torch.Tensor] = {}
		params = list(self.iter_prunable_params(model))
		if not params:
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
		for name, param in params:
			mask = (param.detach().abs() > threshold).to(param.device)
			masks[name] = mask.to(param.dtype)
		self.masks = masks
		return masks