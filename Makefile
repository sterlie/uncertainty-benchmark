.PHONY: run-experiment run-aleatoric run-aleatoric-quick run-epistemic run-epistemic-quick list-methods run-all-methods

# Defaults (override on CLI, e.g. make run-experiment DATASET=chexpert EXPERIMENT=chexpert_gender)
PYTHON       ?= uq/bin/python
DATASET      ?= isic
EXPERIMENT   ?= isic_drop
MODEL 		 ?= densenet
TRAIN_SUBSET ?= 50
TEST_SUBSET  ?= 10
EPOCHS       ?= 5
METHOD       ?= tta

# ── Main entry point ─────────────────────────────────────────────────────
# Usage:
#   make run-experiment DATASET=chexpert EXPERIMENT=chexpert_gender
#   make run-experiment DATASET=chexpert EXPERIMENT=chexpert_age TRAIN_SUBSET=100 TEST_SUBSET=50
#   make run-experiment DATASET=isic EXPERIMENT=isic_drop METHOD=mc_dropout
run-experiment:
	$(PYTHON) -m src.experiments.run_experiment \
		dataset=$(DATASET) \
		experiment=$(EXPERIMENT) \
		model=$(MODEL) \
		method=$(METHOD) \
		dataset.train_subset=$(TRAIN_SUBSET) \
		dataset.test_subset=$(TEST_SUBSET) \
		experiment.epochs=$(EPOCHS) \
		$(ARGS)
