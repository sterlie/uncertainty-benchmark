# A Realistic Medical Imaging Benchmark for Uncertainty Quantification

This repository contains the code for the uncertainty quantification benchmark experiments presented in the paper _A Realistic Medical Imaging Benchmark for Uncertainty Quantification_ ***INSERT LINK*** by ***INSERT AUTHORS***

## Repository Structure

```
uncertainty-benchmark/
├── config/                   # Hydra configuration (structured with config groups)
│   ├── config.yaml           # Root config — composes dataset, experiment, model, method
│   ├── dataset/             # One yaml per dataset (chexpert, isic, mnist, nih, vin)
│   ├── experiment/          # One yaml per experiment (e.g. isic_drop, chexpert_gender)
│   ├── method/              # One yaml per UQ method (e.g. entropy, mc_dropout)
│   ├── model/               # One yaml per model backbone (e.g. DenseNet, mlp)
│   └── optimizer/           # Optimizer configs
├── src/
│   ├── datasets/            # Dataset loaders and torch Dataset classes
│   ├── experiments/
│   │   ├── run_experiment.py # Main Hydra entry point
│   │   ├── tasks.py          # OOD subgroup evaluation task
│   │   └── datasets/        # Per-dataset adapters (isic_adapter, chest_adapter, …)
│   ├── methods/             # UQ method implementations + MethodFactory
│   ├── models/              # Model definitions
│   ├── metrics/             # Evaluation metrics
│   └── utils/               # Shared utilities (visualisation, seeding, etc.)
├── data/                    # Raw and processed datasets
├── models/                  # Saved model checkpoints
├── results/                 # Experiment outputs written by Hydra
├── plots/                   # Generated figures
├── Makefile                  # Convenience targets for running experiments
└── requirements.txt
```

Configuration is managed with [Hydra](https://hydra.cc). The root `config/config.yaml` composes
one config group entry from each of `dataset/`, `experiment/`, `method/`, `model/`, and `optimizer/`.

## Experiment Types

Each experiment config sets `distortion_pattern`, which controls what evaluation is run:

| Pattern type | Example values | What runs |
|---|---|---|
| **Severity distortion** | `blur`, `fracture`, `thinning` | Uncertainty measured per severity level → trend plot + summary JSON |
| **Subgroup shift** | `by_age`, `by_gender`, `by_disease_count`, `age`, `drop`, `hair`, `ink`, `skin_tone` | OOD detection between demographic/attribute subgroups → ROC curves, histograms, line plot, misclassification AUROC |
| **Ambiguity** | `amb` | Uncertainty compared between clearly-labelled and ambiguously-labelled samples |

## Environment Setup

Python 3.10 or newer is required. The environment can be set up using pip or conda.

### Option A — pip + venv

```bash
python -m venv uq
source uq/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Option B — conda

```bash
conda create -n uq python=3.10 -y
conda activate uq

# Install PyTorch first (adjust cuda version as needed, e.g. cu118, cu121, or cpu)
conda install pytorch==2.10.0 torchvision==0.25.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install remaining dependencies
pip install -r requirements.txt
```

> **Note:** `laplace-torch` is a pip-only package and will always be installed via pip regardless of which option you choose.

After installing the main dependencies, install the two bundled local packages in editable mode:

```bash
pip install -e Morpho-MNIST/
pip install -e swa_gaussian/
```

## How to run benchmark

### Check out available methods

List all registered UQ methods:

```bash
ç
```

The available methods are:

| Method | Config key |
|---|---|
| Baseline entropy | `entropy` |
| MC Dropout | `mc_dropout` |
| Test-Time Augmentation | `TTA` |
| Deep Deterministic Uncertainty | `ddu` |
| Deep Ensemble | `ensemble` |
| Heteroscedastic XL | `het_xl` |
| Laplace Approximation | `laplace_approximation` |
| SWAG | `swag` |

Pass `method=all_methods` to run all of the above sequentially in a single invocation.

### Check out available dataset-specific experiments

List all available experiments:

```bash
make list-experiments
```

Experiments are organised by dataset:

| Dataset | Experiment key | Description |
|---|---|---|
| ISIC | `isic_drop` | Dermatoscope dropout artefact |
| ISIC | `isic_hair` | Hair artefact |
| ISIC | `isic_ink` | Ink marker artefact |
| ISIC | `isic_age` | Patient age shift |
| ISIC | `isic_skin_tone` | Skin tone shift |
| MNIST | `mnist_blur` | Gaussian blur |
| MNIST | `mnist_fracture` | Morphological fracture |
| MNIST | `mnist_thinning` | Morphological thinning |
| CheXpert | `chexpert_gender` | Gender subgroup shift |
| CheXpert | `chexpert_age` | Age subgroup shift |
| CheXpert | `chexpert_disease` | Disease label count shift |
| CheXpert | `chexpert_amb` | Ambiguous label detection |
| CheXpert | `chexpert_plain` | No distortion (baseline) |
| NIH | `nih_gender` | Gender subgroup shift |
| NIH | `nih_age` | Age subgroup shift |
| NIH | `nih_disease` | Disease label count shift |
| NIH | `nih_plain` | No distortion (baseline) |
| VinDr | `vin_amb` | Ambiguous label detection |
| VinDr | `vin_disease` | Disease label count shift |
| VinDr | `vin_plain` | No distortion (baseline) |

### Running experiments

**Generic entry point** — works for any dataset/experiment combination:

```bash
make run-experiment DATASET=isic EXPERIMENT=isic_drop METHOD=mc_dropout 
make run-experiment DATASET=isic EXPERIMENT=isic_drop METHOD=all_methods  # run all methods
```

**Dataset shortcuts** — use pre-set defaults for each dataset family:

```bash
# ISIC examples (full dataset)
make run-isic  EXPERIMENT=isic_drop      MODEL=EfficientNet       METHOD=all_methods
make run-isic  EXPERIMENT=isic_hair      MODEL=EfficientNet       METHOD=mc_dropout
make run-isic  EXPERIMENT=isic_ink       MODEL=EfficientNet       METHOD=entropy
make run-isic  EXPERIMENT=isic_age       MODEL=EfficientNet       METHOD=all_methods

# MNIST examples (full dataset)
make run-mnist EXPERIMENT=mnist_blur     MODEL=mlp METHOD=all_methods   OPTIMIZER=sgd
make run-mnist EXPERIMENT=mnist_fracture MODEL=mlp METHOD=all_methods   OPTIMIZER=sgd
make run-mnist EXPERIMENT=mnist_thinning MODEL=mlp METHOD=mc_dropout    OPTIMIZER=sgd

# CheXpert / NIH / VinDr examples (full dataset)
make run-chexpert   EXPERIMENT=chexpert_gender    MODEL=DenseNet      METHOD=all_methods  OPTIMIZER=adam
make run-chexpert   EXPERIMENT=chexpert_age       MODEL=DenseNet      METHOD=all_methods  OPTIMIZER=adam
make run-chexpert   EXPERIMENT=chexpert_disease   MODEL=DenseNet      METHOD=all_methods  OPTIMIZER=adam
make run-chexpert   EXPERIMENT=chexpert_amb       MODEL=DenseNet      METHOD=all_methods  OPTIMIZER=adam
make run-chexpert   EXPERIMENT=chexpert_plain     MODEL=DenseNet      METHOD=all_methods  OPTIMIZER=adam


make run-nih      EXPERIMENT=nih_age         MODEL=DenseNet METHOD=mc_dropout   OPTIMIZER=adam TRAIN_SUBSET=50 TEST_SUBSET=10
make run-vin      EXPERIMENT=vin_amb         MODEL=DenseNet METHOD=entropy      OPTIMIZER=adam

# Run om a subset for quick iteration
make run-chexpert EXPERIMENT=chexpert_gender MODEL=DenseNet METHOD=mc_dropout  OPTIMIZER=adam TRAIN_SUBSET=50 TEST_SUBSET=10
```

**Key overridable parameters:**

| Parameter | Makefile variable | Typical value for chest |
|---|---|---|
| Number of training samples | `TRAIN_SUBSET` | `null` (all data) |
| Number of test samples | `TEST_SUBSET` | `null` (all data) |
| Model backbone | `MODEL` | `DenseNet` |
| UQ method | `METHOD` | `all_methods` |

Any Hydra config key can also be overridden directly via `ARGS`:

```bash
make run-chexpert EXPERIMENT=chexpert_gender ARGS="experiment.lr=1e-4"
```

## Citation 

## Licence

## Contact
