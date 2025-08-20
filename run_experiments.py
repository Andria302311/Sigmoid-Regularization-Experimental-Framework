from __future__ import annotations

import argparse
import os


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, default="/workspace/configs/smoke.yaml")
	parser.add_argument(
		"--allow-duplicate-openmp",
		action="store_true",
		help="Set KMP_DUPLICATE_LIB_OK=TRUE to bypass duplicate OpenMP runtime errors (unsafe).",
	)
	parser.add_argument(
		"--threads",
		type=int,
		default=int(os.environ.get("OMP_NUM_THREADS", "1")),
		help="Max threads for OMP/MKL backends (defaults to 1 if not set).",
	)
	args = parser.parse_args()

	# Configure environment BEFORE importing libraries that may load OpenMP runtimes
	os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
	os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
	if args.allow_duplicate_openmp:
		os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

	from framework.experiments import ExperimentalFramework

	fw = ExperimentalFramework(args.config)
	results = fw.run_experiments()
	fw.generate_reports(results)


if __name__ == "__main__":
	main()