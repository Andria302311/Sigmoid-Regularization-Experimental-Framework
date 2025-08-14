from __future__ import annotations

from typing import Dict, Tuple

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..pruners.base import BasePruner


def train_one_epoch(
	model: nn.Module,
	optimizer: torch.optim.Optimizer,
	criterion: nn.Module,
	loader: DataLoader,
	device: torch.device,
	epoch_idx: int,
	pruner: BasePruner | None = None,
) -> Dict[str, float]:
	model.train()
	t0 = time.time()
	total_loss = 0.0
	count = 0
	for batch in loader:
		x, y = batch
		x = x.to(device)
		y = y.to(device)
		optimizer.zero_grad(set_to_none=True)
		# Infer whether sequence or tabular
		if x.dim() == 3:
			logits = model(x)
		else:
			logits = model(x)
		loss = criterion(logits, y)
		if pruner is not None:
			loss = loss + pruner.compute_sparsity_loss(model, epoch_idx)
		loss.backward()
		optimizer.step()
		total_loss += loss.detach().item()
		count += 1
	elapsed = time.time() - t0
	return {"loss": total_loss / max(count, 1), "time_s": elapsed}


def evaluate_model(
	model: nn.Module,
	criterion: nn.Module,
	loader: DataLoader,
	device: torch.device,
) -> Tuple[float, float]:
	model.eval()
	total_loss = 0.0
	correct = 0
	num = 0
	with torch.no_grad():
		for x, y in loader:
			x = x.to(device)
			y = y.to(device)
			logits = model(x)
			loss = criterion(logits, y)
			total_loss += loss.item()
			num += y.shape[0]
			if logits.ndim == 2 and y.ndim == 1 and logits.shape[1] > 1:
				pred = logits.argmax(dim=1)
				correct += (pred == y).sum().item()
			else:
				# Regression or binary with 1 output: compute MAE-style accuracy proxy
				pred = logits
				correct += 0
		avg_loss = total_loss / max(num, 1)
		acc = (correct / num) if num > 0 else 0.0
	return avg_loss, acc