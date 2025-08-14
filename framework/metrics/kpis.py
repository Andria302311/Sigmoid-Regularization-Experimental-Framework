from __future__ import annotations

from typing import Dict

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _count_params(model: nn.Module) -> int:
	return sum(p.numel() for p in model.parameters())


def _count_zeros(model: nn.Module) -> int:
	return sum(int((p == 0).sum().item()) for p in model.parameters())


def compute_compression_metrics(model: nn.Module, original_num_params: int) -> Dict[str, float]:
	total_params = _count_params(model)
	zero_params = _count_zeros(model)
	effective_params = total_params - zero_params
	compression_ratio = (original_num_params / max(effective_params, 1)) if original_num_params > 0 else 1.0
	return {
		"compression_ratio": float(compression_ratio),
		"sparsity_level": float(zero_params / max(total_params, 1)),
		"model_size_reduction": float((original_num_params - effective_params) / max(original_num_params, 1)),
		"total_params": float(total_params),
		"zero_params": float(zero_params),
	}


def compute_performance_metrics(
	original_score: float,
	pruned_score: float,
	task: str = "classification",
) -> Dict[str, float]:
	if task == "classification":
		accuracy_retention = (pruned_score / max(original_score, 1e-8)) if original_score > 0 else 0.0
		return {"accuracy_retention": float(accuracy_retention)}
	else:
		# For regression lower is better; use inverse ratio
		retention = (original_score / max(pruned_score, 1e-8)) if pruned_score > 0 else 0.0
		return {"loss_retention": float(retention)}


def compute_efficiency_metrics(
	model_before: nn.Module,
	model_after: nn.Module,
	loader: DataLoader,
	device: torch.device,
	num_batches: int = 10,
) -> Dict[str, float]:
	def _bench(model: nn.Module) -> float:
		model.eval()
		t0 = time.time()
		batches = 0
		with torch.no_grad():
			for i, (x, _) in enumerate(loader):
				if i >= num_batches:
					break
				x = x.to(device)
				_ = model(x)
				batches += 1
		elapsed = time.time() - t0
		return elapsed / max(batches, 1)

	try:
		t_before = _bench(model_before)
	except Exception:
		t_before = float("nan")
	try:
		t_after = _bench(model_after)
	except Exception:
		t_after = float("nan")

	# Validate timings and compute speedup
	if (t_before != t_before) or (t_after != t_after):  # NaN checks
		return {"inference_speedup": float("nan")}
	return {"inference_speedup": float(t_before / max(t_after, 1e-8))}