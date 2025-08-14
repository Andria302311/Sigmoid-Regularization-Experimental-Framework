from __future__ import annotations

import argparse

from framework.experiments import ExperimentalFramework


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, default="/workspace/configs/smoke.yaml")
	args = parser.parse_args()
	fw = ExperimentalFramework(args.config)
	results = fw.run_experiments()
	fw.generate_reports(results)


if __name__ == "__main__":
	main()