.PHONY: setup-mnist run-mnist run-mnist-quick run-benchmark run-train run-eval

setup-mnist:
	python -m src.experiments.run_benchmark setup_only=true

run-mnist:
	python -m src.experiments.run_benchmark

run-mnist-quick:
	python -m src.experiments.run_benchmark dataset.train_subset=5000 dataset.test_subset=1000 experiment.epochs=1

run-benchmark:
	python -m src.experiments.run_benchmark

run-train:
	python -m src.experiments.run_train task=train

run-eval:
	python -m src.experiments.run_eval task=eval
