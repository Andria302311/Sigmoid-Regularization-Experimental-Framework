from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class LTCCell(nn.Module):
	"""A simplified discrete-time LTC-like cell with learnable time constants.
	This is not a faithful continuous-time ODE solver but captures adaptive time-scale dynamics.
	"""

	def __init__(self, input_size: int, hidden_size: int, dt: float = 1.0) -> None:
		super().__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.dt = dt
		self.W_in = nn.Linear(input_size, hidden_size, bias=True)
		self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
		self.tau = nn.Parameter(torch.ones(hidden_size))
		self.activation = nn.Tanh()

	def forward(self, x_t: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
		# Ensure positive time constants
		tau = torch.nn.functional.softplus(self.tau) + 1e-3
		dh = (-h_t + self.activation(self.W_in(x_t) + self.W_h(h_t))) / tau
		return h_t + self.dt * dh


class LTCNetwork(nn.Module):
	def __init__(self, input_dim: int, hidden_size: int = 64, output_dim: int = 1) -> None:
		super().__init__()
		self.cell = LTCCell(input_dim, hidden_size)
		self.head = nn.Linear(hidden_size, output_dim)
		self.prunable_parameter_names = {
			name for name, _ in self.named_parameters() if name.endswith("weight")
		}

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: (B, T, D)
		batch, time, feat = x.shape
		h = x.new_zeros(batch, self.cell.hidden_size)
		for t in range(time):
			h = self.cell(x[:, t, :], h)
		return self.head(h)