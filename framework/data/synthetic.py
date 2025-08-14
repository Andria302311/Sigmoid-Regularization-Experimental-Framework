from __future__ import annotations

from typing import Tuple

import math
import torch
from torch.utils.data import Dataset


class ToySequenceClassification(Dataset):
	def __init__(self, num_samples: int = 1024, seq_len: int = 20, input_dim: int = 4, num_classes: int = 2, seed: int = 42):
		rng = torch.Generator()
		rng.manual_seed(seed)
		self.x = torch.randn(num_samples, seq_len, input_dim, generator=rng)
		# Label is sign of sum over a subset of features/time with noise
		signal = self.x[..., 0].sum(dim=1) + 0.5 * self.x[..., 1].sum(dim=1)
		noise = 0.3 * torch.randn(num_samples, generator=rng)
		logits = signal + noise
		self.y = (logits > 0).long() % num_classes
		self.num_classes = num_classes

	def __len__(self) -> int:
		return self.x.shape[0]

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
		return self.x[idx], self.y[idx]


class ToySequenceRegression(Dataset):
	def __init__(self, num_samples: int = 1024, seq_len: int = 30, input_dim: int = 2, seed: int = 0):
		rng = torch.Generator()
		rng.manual_seed(seed)
		t = torch.linspace(0, 2 * math.pi, seq_len)
		w = torch.randn(input_dim, generator=rng)
		x = torch.randn(num_samples, seq_len, input_dim, generator=rng)
		y = (x @ w).sum(dim=2).sin().sum(dim=1, keepdim=True)
		self.x = x
		self.y = y

	def __len__(self) -> int:
		return self.x.shape[0]

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
		return self.x[idx], self.y[idx]


class ToyTabularClassification(Dataset):
	def __init__(self, num_samples: int = 2048, input_dim: int = 16, num_classes: int = 3, seed: int = 7):
		rng = torch.Generator()
		rng.manual_seed(seed)
		self.x = torch.randn(num_samples, input_dim, generator=rng)
		W = torch.randn(input_dim, num_classes, generator=rng)
		logits = self.x @ W
		self.y = logits.argmax(dim=1)
		self.num_classes = num_classes

	def __len__(self) -> int:
		return self.x.shape[0]

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
		return self.x[idx], self.y[idx]