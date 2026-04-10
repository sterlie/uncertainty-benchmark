.PHONY: run-aleatoric run-aleatoric-quick run-epistemic run-epistemic-quick list-methods run-all-methods

# Defaults (override on CLI, e.g. make run-aleatoric-quick DATASET=isic EPOCHS=5)
PYTHON       ?= uq/bin/python
DATASET      ?= isic
MODEL 		 ?= densenet
TRAIN_SUBSET ?= 50
TEST_SUBSET  ?= 10
EPOCHS       ?= 5
METHOD       ?= tta


run-aleatoric:
	$(PYTHON) -m src.experiments.aleatoric_trend dataset=$(DATASET) method=$(METHOD) 

run-aleatoric-quick:
	$(PYTHON) -m src.experiments.aleatoric_trend dataset=$(DATASET) model=$(MODEL) dataset.train_subset=$(TRAIN_SUBSET) dataset.test_subset=$(TEST_SUBSET) experiment.epochs=$(EPOCHS) method=$(METHOD)  $(ARGS)

run-epistemic:
	$(PYTHON) -m src.experiments.epistemic_trend dataset=$(DATASET) experiment=epistemic_trend $(ARGS)

run-epistemic-quick:
	$(PYTHON) -m src.experiments.epistemic_trend dataset=$(DATASET) experiment=epistemic_trend dataset.train_subset=$(TRAIN_SUBSET) dataset.test_subset=$(TEST_SUBSET) experiment.epochs=$(EPOCHS) method=$(METHOD)  $(ARGS)

list-methods:
	$(PYTHON) -c "from src.methods.method_factory import MethodFactory; print('\n'.join(MethodFactory.get_available_methods()))"

run-all-methods:
	$(PYTHON) -m src.experiments.aleatoric_trend dataset=$(DATASET) methods_to_run=all_methods $(ARGS)
