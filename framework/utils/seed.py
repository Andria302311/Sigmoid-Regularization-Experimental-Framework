from __future__ import annotations

import os
import random
import numpy as np
import torch


def set_global_seed(seed: int = 42, deterministic_torch: bool = True) -> None:
	random.seed(seed)
	np.random.seed(seed)
	os.environ["PYTHONHASHSEED"] = str(seed)
	try:
		torch.use_deterministic_algorithms(deterministic_torch)
	except Exception:
		pass
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)