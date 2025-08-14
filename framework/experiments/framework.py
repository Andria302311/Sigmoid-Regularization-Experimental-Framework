from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from ..models import PrunableMLP, PrunableLSTM, PrunableGRU, LTCNetwork
from ..pruners.sigmoid import SigmoidRegularizationPruner
from ..pruners.magnitude import MagnitudePruner
from ..pruners.random import RandomPruner
from ..pruners.l1 import L1RegularizationPruner
from ..data import ToySequenceClassification, ToySequenceRegression, ToyTabularClassification
from ..utils import set_global_seed, train_one_epoch, evaluate_model
from ..metrics import compute_compression_metrics, compute_performance_metrics, compute_efficiency_metrics


@dataclass
class ExperimentalResults:
	original_score: float
	pruned_score: float
	compression: Dict[str, float]
	efficiency: Dict[str, float]
	performance: Dict[str, float]


class ExperimentalFramework:
	def __init__(self, config_path: str) -> None:
		with open(config_path, "r") as f:
			self.cfg: Dict[str, Any] = yaml.safe_load(f)
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		set_global_seed(self.cfg.get("random_seed", 42))

	def setup_datasets(self) -> Dict[str, DataLoader]:
		ds_cfg = self.cfg.get("dataset", {"name": "toy_seq_cls"})
		name = ds_cfg.get("name", "toy_seq_cls")
		batch_size = int(self.cfg.get("training", {}).get("batch_size", 64))
		if name == "toy_seq_cls":
			dataset = ToySequenceClassification(num_samples=ds_cfg.get("num_samples", 1024), seq_len=ds_cfg.get("seq_len", 20), input_dim=ds_cfg.get("input_dim", 4), num_classes=ds_cfg.get("num_classes", 2))
			input_dim = ds_cfg.get("input_dim", 4)
			output_dim = ds_cfg.get("num_classes", 2)
			self.task_type = "classification"
		elif name == "toy_seq_reg":
			dataset = ToySequenceRegression(num_samples=ds_cfg.get("num_samples", 1024), seq_len=ds_cfg.get("seq_len", 30), input_dim=ds_cfg.get("input_dim", 2))
			input_dim = ds_cfg.get("input_dim", 2)
			output_dim = 1
			self.task_type = "regression"
		elif name == "toy_tabular_cls":
			dataset = ToyTabularClassification(num_samples=ds_cfg.get("num_samples", 2048), input_dim=ds_cfg.get("input_dim", 16), num_classes=ds_cfg.get("num_classes", 3))
			input_dim = ds_cfg.get("input_dim", 16)
			output_dim = ds_cfg.get("num_classes", 3)
			self.task_type = "classification"
		else:
			raise ValueError(f"Unknown dataset {name}")
		self.input_dim = input_dim
		self.output_dim = output_dim
		# Split
		val_ratio = float(self.cfg.get("dataset", {}).get("val_ratio", 0.2))
		val_size = int(len(dataset) * val_ratio)
		train_size = len(dataset) - val_size
		train_ds, val_ds = random_split(dataset, [train_size, val_size])
		train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
		val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
		return {"train": train_loader, "val": val_loader}

	def setup_models(self) -> Dict[str, nn.Module]:
		m_cfg = self.cfg.get("model", {"name": "mlp"})
		name = m_cfg.get("name", "mlp")
		if name == "mlp":
			model = PrunableMLP(
				input_dim=self.input_dim if self.task_type != "classification" or len(self.cfg.get("dataset", {}).get("name", "toy_seq_cls")) == 0 else self.input_dim,
				output_dim=self.output_dim,
				hidden_sizes=m_cfg.get("hidden_sizes", [128, 128]),
				activation=m_cfg.get("activation", "ReLU"),
			)
		elif name == "lstm":
			model = PrunableLSTM(input_dim=self.input_dim, hidden_size=m_cfg.get("hidden_size", 64), num_layers=m_cfg.get("num_layers", 1), bidirectional=m_cfg.get("bidirectional", False), output_dim=self.output_dim)
		elif name == "gru":
			model = PrunableGRU(input_dim=self.input_dim, hidden_size=m_cfg.get("hidden_size", 64), num_layers=m_cfg.get("num_layers", 1), bidirectional=m_cfg.get("bidirectional", False), output_dim=self.output_dim)
		elif name == "ltc":
			model = LTCNetwork(input_dim=self.input_dim, hidden_size=m_cfg.get("hidden_size", 64), output_dim=self.output_dim)
		else:
			raise ValueError(f"Unknown model {name}")
		return {"main": model.to(self.device)}

	def setup_pruners(self) -> Dict[str, BaseException]:
		p_cfg = self.cfg.get("pruner", {"name": "sigmoid"})
		name = p_cfg.get("name", "sigmoid")
		if name == "sigmoid":
			pruner = SigmoidRegularizationPruner(
				epsilon=p_cfg.get("epsilon", 0.05),
				alpha=p_cfg.get("alpha", 0.01),
				a=p_cfg.get("a", 0.99),
				pruning_threshold=p_cfg.get("pruning_threshold", 0.05),
				target_sparsity=p_cfg.get("target_sparsity", None),
				warmup_epochs=p_cfg.get("warmup_epochs", 5),
			)
		elif name == "magnitude":
			pruner = MagnitudePruner(target_sparsity=p_cfg.get("target_sparsity", 0.5), global_pruning=p_cfg.get("global", True))
		elif name == "random":
			pruner = RandomPruner(target_sparsity=p_cfg.get("target_sparsity", 0.5), seed=self.cfg.get("random_seed", 42))
		elif name == "l1":
			pruner = L1RegularizationPruner(lambda_l1=p_cfg.get("lambda", 1e-4), target_sparsity=p_cfg.get("target_sparsity", None), pruning_threshold=p_cfg.get("pruning_threshold", None))
		else:
			raise ValueError(f"Unknown pruner {name}")
		return {"main": pruner}

	def run_experiments(self) -> ExperimentalResults:
		loaders = self.setup_datasets()
		models = self.setup_models()
		pruners = self.setup_pruners()
		model = models["main"]
		pruner = pruners["main"]

		if self.task_type == "classification":
			criterion = nn.CrossEntropyLoss()
		else:
			criterion = nn.MSELoss()
		optimizer = torch.optim.Adam(model.parameters(), lr=float(self.cfg.get("training", {}).get("lr", 1e-3)))

		epochs = int(self.cfg.get("training", {}).get("epochs", 5))
		train_losses = []
		sparsities = []
		for epoch in range(epochs):
			metrics = train_one_epoch(model, optimizer, criterion, loaders["train"], self.device, epoch, pruner)
			val_loss, val_acc = evaluate_model(model, criterion, loaders["val"], self.device)
			train_losses.append(metrics["loss"])
			sparsities.append(self._current_sparsity(model))

		# Baseline score before pruning masks applied
		orig_loss, orig_acc = evaluate_model(model, criterion, loaders["val"], self.device)
		original_score = orig_acc if self.task_type == "classification" else orig_loss

		# Create and apply masks
		pruner.create_pruning_masks(model)
		pruner.apply_masks(model)

		# Evaluate after pruning
		pruned_loss, pruned_acc = evaluate_model(model, criterion, loaders["val"], self.device)
		pruned_score = pruned_acc if self.task_type == "classification" else pruned_loss

		compression = compute_compression_metrics(model, original_num_params=sum(p.numel() for p in model.parameters()))
		performance = compute_performance_metrics(original_score, pruned_score, task=self.task_type)
		try:
			efficiency = compute_efficiency_metrics(model, model, loaders["val"], self.device, num_batches=5)
		except Exception:
			efficiency = {"inference_speedup": float("nan")}

		return ExperimentalResults(
			original_score=float(original_score),
			pruned_score=float(pruned_score),
			compression=compression,
			efficiency=efficiency,
			performance=performance,
		)

	def generate_reports(self, results: ExperimentalResults) -> None:
		# Placeholder: could serialize to disk, W&B, or console
		print({
			"original_score": results.original_score,
			"pruned_score": results.pruned_score,
			"compression": results.compression,
			"efficiency": results.efficiency,
			"performance": results.performance,
		})

	def _current_sparsity(self, model: nn.Module) -> float:
		zero = sum(int((p == 0).sum().item()) for p in model.parameters())
		total = sum(p.numel() for p in model.parameters())
		return float(zero / max(total, 1))