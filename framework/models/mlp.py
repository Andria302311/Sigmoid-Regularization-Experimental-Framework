from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn


class PrunableMLP(nn.Module):
	def __init__(
		self,
		input_dim: int,
		output_dim: int,
		hidden_sizes: Sequence[int] = (128, 128),
		activation: str = "ReLU",
		dropout: float = 0.0,
	) -> None:
		super().__init__()
		layers: List[nn.Module] = []
		prev = input_dim
		act_layer = getattr(nn, activation)
		for hidden in hidden_sizes:
			layers.append(nn.Linear(prev, hidden))
			layers.append(act_layer())
			if dropout > 0:
				layers.append(nn.Dropout(dropout))
			prev = hidden
		layers.append(nn.Linear(prev, output_dim))
		self.net = nn.Sequential(*layers)

		self.prunable_parameter_names = {
			name for name, _ in self.named_parameters() if name.endswith("weight")
		}

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)