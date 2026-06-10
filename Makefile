.PHONY: run-experiment list-methods list-experiments \
        run-isic run-mnist \
        run-chexpert run-nih run-vin

# Defaults (override on CLI, e.g. make run-experiment DATASET=chexpert EXPERIMENT=chexpert_gender)
PYTHON       ?= uq/bin/python
DATASET      ?= isic
EXPERIMENT   ?= isic_drop
MODEL        ?= mlp
TRAIN_SUBSET ?= null
TEST_SUBSET  ?= null
METHOD       ?= tta
OPTIMIZER    ?= sgd_derma

# ── Utilities ─────────────────────────────────────────────────────────────
list-methods:
	$(PYTHON) -c "from src.methods import MethodFactory; [print(m) for m in sorted(MethodFactory.get_available_methods())]"

list-experiments:
	@ls config/experiment/*.yaml | xargs -I{} basename {} .yaml | sort

# ── Main entry point ─────────────────────────────────────────────────────
# Usage:
#   make run-experiment DATASET=chexpert EXPERIMENT=chexpert_gender
#   make run-experiment DATASET=chexpert EXPERIMENT=chexpert_age TRAIN_SUBSET=100 TEST_SUBSET=50
#   make run-experiment DATASET=isic EXPERIMENT=isic_drop METHOD=mc_dropout
#   make run-experiment DATASET=isic EXPERIMENT=isic_drop METHOD=all_methods
run-experiment:
	$(PYTHON) -m src.experiments.run_experiment \
		dataset=$(DATASET) \
		experiment=$(EXPERIMENT) \
		model=$(MODEL) \
		method=$(METHOD) \
		dataset.train_subset=$(TRAIN_SUBSET) \
		dataset.test_subset=$(TEST_SUBSET) \
		$(ARGS)

# ── Quick isic / mnist experiments ──────────────────────────────────────
# Usage:
#   make run-isic  EXPERIMENT=isic_drop
#   make run-isic  EXPERIMENT=isic_hair METHOD=mc_dropout
#   make run-mnist EXPERIMENT=mnist_blur METHOD=entropy
run-isic:
	$(PYTHON) -m src.experiments.run_experiment \
		dataset=isic \
		experiment=$(EXPERIMENT) \
		model=$(MODEL) \
		method=$(METHOD) \
		optimizer=$(OPTIMIZER) \
		dataset.train_subset=$(TRAIN_SUBSET) \
		dataset.test_subset=$(TEST_SUBSET) \
		$(ARGS)

run-mnist:
	$(PYTHON) -m src.experiments.run_experiment \
		dataset=mnist \
		experiment=$(EXPERIMENT) \
		model=$(MODEL) \
		method=$(METHOD) \
		optimizer=$(OPTIMIZER) \
		dataset.train_subset=$(TRAIN_SUBSET) \
		dataset.test_subset=$(TEST_SUBSET) \
		$(ARGS)

# ── Quick chest experiments ───────────────────────────────────────────────
# Usage:
#   make run-chexpert EXPERIMENT=chexpert_gender MODEL=DenseNet METHOD=all_methods TRAIN_SUBSET=100 TEST_SUBSET=50
#   make run-chexpert EXPERIMENT=chexpert_age    MODEL=DenseNet METHOD=ddu         TRAIN_SUBSET=100 TEST_SUBSET=50
#   make run-chexpert EXPERIMENT=chexpert_amb    MODEL=DenseNet METHOD=entropy     TRAIN_SUBSET=100 TEST_SUBSET=50
#   make run-nih      EXPERIMENT=nih_gender      MODEL=DenseNet METHOD=all_methods TRAIN_SUBSET=100 TEST_SUBSET=50
#   make run-nih      EXPERIMENT=nih_age         MODEL=DenseNet METHOD=mc_dropout  TRAIN_SUBSET=100 TEST_SUBSET=50
#   make run-vin      EXPERIMENT=vin_amb         MODEL=DenseNet METHOD=entropy     TRAIN_SUBSET=100 TEST_SUBSET=50
run-chexpert:
	$(PYTHON) -m src.experiments.run_experiment \
		dataset=chexpert \
		experiment=$(EXPERIMENT) \
		model=$(MODEL) \
		method=$(METHOD) \
		optimizer=$(OPTIMIZER) \
		dataset.train_subset=$(TRAIN_SUBSET) \
		dataset.test_subset=$(TEST_SUBSET) \
		$(ARGS)

run-nih:
	$(PYTHON) -m src.experiments.run_experiment \
		dataset=nih \
		experiment=$(EXPERIMENT) \
		model=$(MODEL) \
		optimizer=$(OPTIMIZER) \
		method=$(METHOD) \
		dataset.train_subset=$(TRAIN_SUBSET) \
		dataset.test_subset=$(TEST_SUBSET) \
		$(ARGS)

run-vin:
	$(PYTHON) -m src.experiments.run_experiment \
		dataset=vin \
		experiment=$(EXPERIMENT) \
		model=$(MODEL) \
		method=$(METHOD) \
		optimizer=$(OPTIMIZER) \
		dataset.train_subset=$(TRAIN_SUBSET) \
		dataset.test_subset=$(TEST_SUBSET) \
		$(ARGS)
