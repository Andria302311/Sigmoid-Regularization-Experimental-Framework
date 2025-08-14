# Sigmoid Regularization Experimental Framework

Quick start:

1. Install dependencies:
   ```bash
   pip install -r /workspace/requirements.txt
   ```
2. Run a smoke test experiment:
   ```bash
   python /workspace/run_experiments.py --config /workspace/configs/smoke.yaml
   ```

The framework provides pruners (SigmoidRegularizationPruner, MagnitudePruner, RandomPruner, L1RegularizationPruner), prunable models (MLP, LSTM, GRU), a simple LTC baseline, synthetic datasets for quick tests, KPI metrics, and a config-driven experiment manager.
