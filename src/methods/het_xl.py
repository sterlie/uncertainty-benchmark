import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf

from src.methods.method import Method
from src.methods.method_factory import register_method
from src.models.model_factory import ModelFactory


def get_last_linear_layer(model):
    last_linear = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_linear = module
            last_name = name
    return last_name, last_linear

@register_method("het_xl")
class HetXL(Method):
    def hook(self, module, input, output):
        self.features = output.detach()

    def __init__(self, config):
        self.sample_size = config.method.get('sample_size', 100)
        self.matrix_rank = config.method.get('matrix_rank', 16)
        self.temperature = config.method.get('temperature', 1.0)
        self.use_het = config.method.get('use_het', False)
        self.num_classes = config.dataset.get('num_classes', 10)
        # use_het=True  → samples in logit space (num_classes dims)
        # use_het=False → samples in feature space (hidden_dim dims); output_features
        #                 config value is ignored to avoid shape mismatches.
        if self.use_het:
            self.output_features = self.num_classes
        else:
            self.output_features = config.model.get('hidden_dim', config.method.get('output_features', 512))
        super(HetXL, self).__init__(config)
        self.ood_threshold = config.method.get('ood_threshold', 0.0)
        self.misclassify_threshold = config.method.get('misclassify_threshold', self.ood_threshold)
        self.model_dir = "het_xl"

        self.named_modules = dict(self.model.named_modules())
        self.key = list(self.named_modules.keys())[-2]
        self.features = None

    def init_model(self):
        """Initialize the model. Must be implemented by child classes."""
        self.model = ModelFactory.create(self.config)
        self._low_rank_cov_layer = nn.Linear(
            in_features=self.config.model.hidden_dim,
            out_features=self.output_features * self.matrix_rank,
        )
        self._diagonal_std_layer = nn.Linear(
            in_features=self.config.model.hidden_dim,
            out_features=self.output_features,
        )
        self._min_scale_monte_carlo = 1e-3
        self.model.to(self.device)
        self._low_rank_cov_layer.to(self.device)
        self._diagonal_std_layer.to(self.device)

    def init_optimizer(self):
        """Initialize the optimizer. Must be implemented by child classes."""
        if self.config.optimizer.name == "SGD":
            optimizer_class = torch.optim.SGD
        elif self.config.optimizer.name == "Adam":
            optimizer_class = torch.optim.Adam
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer.name}")
        arguments = OmegaConf.to_container(self.config.optimizer)
        arguments.pop("name")
        arguments.pop("epochs")
        scheduler_arguments = arguments.pop("scheduler", None)
        self.optimizer = optimizer_class(
            [{"params": self.model.parameters()}, {"params": self._low_rank_cov_layer.parameters()}, {"params": self._diagonal_std_layer.parameters()}],
            **arguments,
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, scheduler_arguments["milestones"], scheduler_arguments["gamma"]) if scheduler_arguments is not None else None

    def build_base_model(self, retrain=False, **kwargs):
        pass

    def run_model(self, inputs: torch.Tensor, return_mean=True, return_variance=False):
        self.handle = self.named_modules[self.key].register_forward_hook(self.hook)

        outputs = self.model(inputs)
        features = self.model.features if hasattr(self.model, "features") else self.features
        B = features.shape[0]
        D_out = self.output_features
        R = self.matrix_rank
        S = self.sample_size

        low_rank_cov = self._low_rank_cov_layer(features).reshape(-1, D_out, R)  # [B, C | D, R]
        diagonal_std = (F.softplus(self._diagonal_std_layer(features)) + self._min_scale_monte_carlo)  # [B, C | D]

        diagonal_samples = diagonal_std.unsqueeze(1) * torch.randn(B, S, D_out, device=features.device)  # [B, S, C | D]
        standard_samples = torch.randn(B, S, R, device=features.device)  # [B, S, R]
        einsum_res = torch.einsum("bdr, bsr -> bsd", low_rank_cov, standard_samples)  # [B, S, C | D]
        samples = einsum_res + diagonal_samples  # [B, S, C | D]

        name, last_layer = get_last_linear_layer(self.model)

        if self.use_het:
            logits = last_layer(features)  # [B, C]
            logits = logits.unsqueeze(1) + samples  # [B, S, C]
        else:
            pre_logits = features.unsqueeze(1) + samples  # [B, S, D]
            logits = last_layer(pre_logits)

        logits_temperature = logits / self.temperature
        variance = diagonal_std
        # variance.mean(dim=1).unsqueeze(1).repeat(1, logits.shape[-1])
        self.handle.remove()
        if return_mean:
            if return_variance:
                return logits_temperature.mean(dim=1), variance
            return logits_temperature.mean(dim=1)
        if return_variance:
            return logits_temperature.transpose(0, 1), variance
        return logits_temperature.transpose(0, 1)


    def train_uncertainty_method(self, train_loader, valid_loader, loss_weight=None):
        optimizer = self.optimizer
        criterion = nn.CrossEntropyLoss(loss_weight)

        # Training

        best_acc = 0.0
        best_previous_loss = float('inf')
        epochs = self.config.optimizer.get('epochs', 10)
        dataset_name = self.config.dataset.name
        model_name = self.config.model.name
        _best_ckpt = Path(os.getcwd()) / "models" / dataset_name / "het_xl" / f"best_model_{model_name}.pt"
        _best_ckpt.parent.mkdir(parents=True, exist_ok=True)
        print("Any trainable params:",
              any(p.requires_grad for p in self.model.parameters()))
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            total_correct = 0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.run_model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_correct += (outputs.argmax(1) == targets).sum().item()

            avg_loss = total_loss / len(train_loader)

            # -------- VALIDATION --------
            self.model.eval()
            val_loss = 0.0
            val_correct = 0

            with torch.no_grad():
                for inputs, targets in valid_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    outputs = self.run_model(inputs)
                    # print(outputs, targets)
                    loss = criterion(outputs, targets)

                    val_loss += loss.item()
                    val_correct += (outputs.argmax(1) == targets).sum().item()

            avg_val_loss = val_loss / len(valid_loader)
            val_acc = val_correct / len(valid_loader.dataset)

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), _best_ckpt)

            print(
                f'Epoch {epoch + 1}/{epochs} - Train: Loss: {avg_loss:.4f}, Accuracy: {total_correct / len(train_loader.dataset):.4f}\nValidation: Loss {avg_val_loss:.4f}, Accuracy: {val_acc:.4f}')

        if _best_ckpt.exists():
            self.model.load_state_dict(torch.load(_best_ckpt, map_location=self.device, weights_only=True))
            _best_ckpt.unlink()

        return self.model

    def inference(self, loader: torch.utils.data.DataLoader):
        self.model.eval()

        predictions = []
        labels = []
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                preds, variance = self.run_model(inputs, return_mean=False, return_variance=True)
                preds = F.softmax(preds, dim=2)
                predictions.append(preds)
                labels.append(targets)

        predictions = torch.cat(predictions, dim=1)
        labels = torch.cat(labels, dim=0)
        return predictions, labels
