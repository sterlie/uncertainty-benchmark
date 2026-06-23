import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from sympy.printing.pretty.pretty_symbology import pretty_atom
from torch.utils.data import DataLoader

from src.methods.method import Method
from src.methods.method_factory import register_method
from src.models.model_factory import ModelFactory


@register_method("ensemble")
class Ensemble(Method):
    """Ensemble for uncertainty quantification.

    This method uses multiple models to approximate Bayesian inference.
    """

    def __init__(self, config):
        self.sample_size = config.method.get('sample_size', 1)
        super(Ensemble, self).__init__(config)
        self.model_dir = config.method.name
        self.ood_threshold = config.method.get('ood_threshold', 1.0)
        self.misclassify_threshold = config.method.get('misclassify_threshold', self.ood_threshold)

    def init_model(self):
        """Initialize the model. Must be implemented by child classes."""
        self.model = [ModelFactory.create(self.config) for _ in range(self.sample_size)]
        for model in self.model:
            model.to(self.device)
        return

    def init_optimizer(self):
        """Initialize the optimizer. Must be implemented by child classes."""
        optimizer_name = str(self.config.optimizer.name).lower()
        if optimizer_name.startswith("sgd"):
            optimizer_class = torch.optim.SGD
        elif optimizer_name.startswith("adam"):
            optimizer_class = torch.optim.Adam
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer.name}")
        arguments = OmegaConf.to_container(self.config.optimizer)
        arguments.pop("name")
        arguments.pop("epochs")
        arguments.pop("early_stopping_patience", None)
        arguments.pop("early_stopping_min_delta", None)
        arguments.pop("early_stopping_monitor", None)
        arguments.pop("scheduler", None)
        self.optimizer = [optimizer_class(
            model.parameters(),
            **arguments,
        ) for model in self.model]

    def load_pretrained_model(self, pretrained):
        print("loading ", pretrained)
        if (isinstance(pretrained, Path) and pretrained.exists()) or (isinstance(pretrained, str) and os.path.exists(pretrained)):
            for model, state_dict in zip(self.model, torch.load(pretrained, weights_only=True, map_location=self.device)):
                model.load_state_dict(state_dict)
            print(f"Loaded pretrained model from {pretrained}")
            return True
        return False

    def save_pretrained_model(self, pretrained):
        torch.save([model.state_dict() for model in self.model], str(pretrained))
        print(f"Saved model to {pretrained}")

    def build_base_model(self, retrain=False, **kwargs):
        pass

    def build_method(self, rebuild=False, **kwargs):
        if not rebuild:
            if 'pretrained' in kwargs:
                pretrained_model = os.path.join(self.config.output.path, self.config.method.name, 'model', os.path.basename(kwargs['pretrained']))
            else:
                pretrained_model = os.path.join(self.config.output.path, self.config.method.name, 'model', 'models.pt')
            if self.load_pretrained_model(pretrained_model):
                return

        if self.config.weighted:
            if 'weights' in self.config.dataset:
                weights = torch.tensor(self.config.dataset.weights).float().to(self.device)
            else:
                labels = torch.tensor(kwargs['train_loader'].dataset.labels)
                class_counts = torch.bincount(labels, minlength=self.num_classes)
                class_counts[class_counts == 0] = 1

                N = labels.size(0)
                weights = N / (self.num_classes * class_counts.float())
                weights = weights.to(self.device)
            print("weights", weights)
            self.train_uncertainty_method(kwargs['train_loader'], kwargs['valid_loader'], weights)
        else:
            self.train_uncertainty_method(kwargs['train_loader'], kwargs['valid_loader'])
        output_save_dir = os.path.join(self.config.output.path, self.config.method.name, 'model')
        os.makedirs(output_save_dir, exist_ok=True)
        if 'model_name' in kwargs:
            filename = os.path.join(output_save_dir,kwargs['model_name'])
        else:
            filename = os.path.join(output_save_dir, 'models.pt')
        self.save_pretrained_model(filename)


    def train_uncertainty_method(self, train_loader, valid_loader, loss_weight=None):
        """Train the model using standard supervised learning.

        Args:
            train_loader: Training data loader
            test_loader: Test data loader
            retrain: Whether to retrain the model
        """

        # Setup optimizer
        criterion = nn.CrossEntropyLoss(loss_weight)

        # Training
        epochs = self.config.optimizer.get('epochs', 10)
        for model in self.model:
            model.train()

        for epoch in range(epochs):
            total_loss = 0.0
            total_correct = 0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                for model, optimizer in zip(self.model, self.optimizer):
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() / self.sample_size
                    total_correct += (outputs.argmax(1) == targets).sum().item() / self.sample_size

            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {total_correct / len(train_loader.dataset):.4f}')

        # Evaluation
        for model in self.model:
            model.eval()
        total_loss = 0.0
        total_correct = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valid_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                for model in self.model:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    total_loss += loss.item() / self.sample_size
                    total_correct += (outputs.argmax(1) == targets).sum().item() / self.sample_size

        avg_loss = total_loss / len(valid_loader)
        print(f'Test Loss: {avg_loss:.4f}, Accuracy: {total_correct / len(valid_loader.dataset):.4f}')

    def predict(self, inputs: torch.Tensor):
        """Make predictions in a conventional manner.

        Args:
            inputs: Input tensor

        Returns:
            The predictions of the model
        """
        inputs = inputs.to(self.device)
        for model in self.model:
            model.eval()

        predictions = torch.mean(torch.stack([model(inputs) for model in self.model]), dim=0)
        return predictions

    def measure_uncertainty(self, test_loader: torch.utils.data.DataLoader):
        results = dict()
        for inputs, targets in test_loader:
            result = self.do_measure_uncertainty(inputs, targets)
            for key, value in result.items():
                results[key] = torch.cat([results[key], value]) if key in results else value
        return results

    def do_measure_uncertainty(self, inputs: torch.Tensor, targets: torch.Tensor):
        """Measure uncertainty. Must be implemented by child classes.

        Returns:
            Dictionary containing uncertainty measures:
                - total_uncertainty
                - aleatoric_uncertainty (data uncertainty)
                - epistemic_uncertainty (model uncertainty)
                - out_of_distribution (OOD score)
        """
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        for model in self.model:
            model.eval()

        with torch.no_grad():
            all_logits = torch.stack([model(inputs) for model in self.model])
            all_probs = F.softmax(all_logits, dim=-1)

            predictive_probs = all_probs.mean(dim=0)
            predictive_entropy = -torch.sum(predictive_probs * torch.log(predictive_probs + self.eps), axis=-1)
            expected_entropy = -torch.mean(torch.sum(all_probs * torch.log(all_probs + self.eps), axis=-1), dim=0)
            epistemic = predictive_entropy - expected_entropy
            mutual_information = (all_probs * (torch.log(all_probs + self.eps) - torch.log(all_probs + self.eps))).sum(dim=2).mean(
                dim=0)  # [N]
            ood_score = predictive_entropy
            misclassify_score = predictive_entropy

            var_epistemic = all_probs.var(dim=0).sum(dim=-1)  # [B, C] --> [B]
            var_aleatoric = (all_probs * (1 - all_probs)).mean(dim=0).sum(dim=-1)
            var_total = var_epistemic + var_aleatoric

        return {
            "predictions": predictive_probs,
            "predicted_labels": predictive_probs.argmax(dim=-1),
            "ground_truth": targets,
            "total_uncertainty": predictive_entropy,
            "aleatoric_uncertainty": expected_entropy,
            "epistemic_uncertainty": epistemic,
            "mutual_information": mutual_information,
            "variance_epistemic_uncertainty": var_epistemic,
            "variance_aleatoric_uncertainty": var_aleatoric,
            "variance_total_uncertainty": var_total,
        }
