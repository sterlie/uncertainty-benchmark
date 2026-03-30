

private notes to understand what is going on: 

Set up: 
1) Train model on clean data. 
    Datasets: mnist, chest X-ray (CheXpert, NIH, VinDr-CXR), dermatology (MILK10k). 
2) Apply UQ method: 
    probibalistic methods: MC dropout, deep ensemble, SWAG, laplace, test-time augmentation, HETXL. 
    Deterministic: Entropy, DDU. 
2) Create contolled condition where uncertainty is expected to increase.  

To test Aletoric uncertainty: 
MNIST -> data distortions 
chest xray -> young/old, patient have more dieseases . 
Dermatology -> age group, underrepresetned skintones. 

To test Epistemic uncertainty: 
MNIST -> blur images.
chest xray -> radiologist disagreement
Dermatology -> poor image quality, comments of 'gel', 'water drop', or 'dermoscopy liquid'. 

Disentanflement of uncertainty: AUROC curve comparison. 
------------------------------------------------------------------------------------------------



# Uncertainty Benchmark (MNIST starter)

This repo now has a Hydra-based, reproducible MNIST experiment flow.

## 1) Install

```bash
pip install -r requirements.txt
```

## 2) One-line commands

### Unified core CLI (primary interface)
```bash
python -m src.experiments.run_benchmark dataset=mnist method=mc_dropout model=mlp experiment=aleatoric_trend task=benchmark
```

### Train-only and eval-only entrypoints
```bash
python -m src.experiments.run_train dataset=mnist method=mc_dropout model=mlp experiment=aleatoric_trend task=train
python -m src.experiments.run_eval dataset=mnist method=mc_dropout model=mlp experiment=aleatoric_trend task=eval
```

### Quick smoke test
```bash
python -m src.experiments.run_benchmark dataset=mnist experiment=aleatoric_trend  dataset.train_subset=500 dataset.test_subset=100 experiment.epochs=1
```

## 3) Makefile shortcuts

```bash
make setup-mnist
make run-mnist
make run-mnist-quick
make run-benchmark
make run-train
make run-eval
```

All Makefile shortcuts now resolve to `src.experiments.*` entrypoints.

## 4) Config structure (Hydra)

- `config/config.yaml`
- `config/dataset/` (`mnist`, `chexpert`, `nih`, `vin`, `isic`)
- `config/experiment/` (`mnist_baseline`, `aleatoric_trend`)
- `config/task/` (`benchmark`, `train`, `eval`)

Override any setting from CLI, for example:

```bash
python -m src.experiments.run_benchmark dataset=nih seed=123 experiment.epochs=10 dataset.batch_size=8
```

## 5) Reproducibility

Each run writes outputs to a timestamped Hydra directory under `results/`, including:

- `metrics.json` (history + resolved config + environment)
- `model.pt` (if enabled)

Data is local-only for now (no DVC).