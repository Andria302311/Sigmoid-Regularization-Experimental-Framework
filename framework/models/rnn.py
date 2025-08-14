from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class _BasePrunableRNN(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self._set_prunable_names()

	def _set_prunable_names(self) -> None:
		self.prunable_parameter_names = {
			name for name, _ in self.named_parameters() if "weight" in name
		}


class PrunableLSTM(_BasePrunableRNN):
	def __init__(
		self,
		input_dim: int,
		hidden_size: int = 128,
		num_layers: int = 1,
		bidirectional: bool = False,
		output_dim: int | None = None,
	) -> None:
		super().__init__()
		self.lstm = nn.LSTM(
			input_size=input_dim,
			hidden_size=hidden_size,
			num_layers=num_layers,
			batch_first=True,
			bidirectional=bidirectional,
		)
		directions = 2 if bidirectional else 1
		self.output_dim = output_dim or hidden_size * directions
		self.head = nn.Linear(hidden_size * directions, self.output_dim)
		self._set_prunable_names()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: (B, T, D)
		out, _ = self.lstm(x)
		# Use last time step
		last = out[:, -1, :]
		return self.head(last)


class PrunableGRU(_BasePrunableRNN):
	def __init__(
		self,
		input_dim: int,
		hidden_size: int = 128,
		num_layers: int = 1,
		bidirectional: bool = False,
		output_dim: int | None = None,
	) -> None:
		super().__init__()
		self.gru = nn.GRU(
			input_size=input_dim,
			hidden_size=hidden_size,
			num_layers=num_layers,
			batch_first=True,
			bidirectional=bidirectional,
		)
		directions = 2 if bidirectional else 1
		self.output_dim = output_dim or hidden_size * directions
		self.head = nn.Linear(hidden_size * directions, self.output_dim)
		self._set_prunable_names()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		out, _ = self.gru(x)
		last = out[:, -1, :]
		return self.head(last)