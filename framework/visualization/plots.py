from __future__ import annotations

from typing import List, Tuple

import matplotlib.pyplot as plt


def plot_pareto_frontier(points: List[Tuple[float, float]] | None = None):
	plt.figure()
	if points:
		x, y = zip(*points)
		plt.scatter(x, y)
	plt.xlabel("Sparsity")
	plt.ylabel("Performance")
	plt.title("Sparsity vs Performance")
	plt.close()


def plot_loss_and_sparsity_evolution(losses: List[float] | None = None, sparsities: List[float] | None = None):
	plt.figure()
	if losses:
		plt.plot(losses, label="loss")
	if sparsities:
		plt.plot(sparsities, label="sparsity")
	plt.legend()
	plt.title("Training Dynamics")
	plt.close()