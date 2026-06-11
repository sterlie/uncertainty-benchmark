from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from omegaconf import OmegaConf
import os
from pathlib import Path
from src.models.model_factory import ModelFactory
import numpy as np
from torch.nn import functional as F

from src.utils.metrics import get_acc_score, get_auc_score, get_NLL_score


class Method(ABC):
    """Base method class for all uncertainty quantification methods."""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eps = 1e-10
        self.num_classes = config.dataset.num_classes
        self.init_model()
        self.init_optimizer()

    def init_model(self):
        """Initialize the model. Must be implemented by child classes."""
        self.model = ModelFactory.create(self.config)
        self.model.to(self.device)

    def init_optimizer(self):
        """Initialize the optimizer. Must be implemented by child classes."""
        optimizer_name = str(self.config.optimizer.name).lower()
        cfg = self.config.optimizer
        if optimizer_name.startswith("sgd"):
            self.optimizer =  torch.optim.SGD(
                self.model.parameters(),
                lr=cfg.lr,
                momentum=cfg.get("momentum", 0.0),
                weight_decay=cfg.get("weight_decay", 0.0),
            )
        elif optimizer_name.startswith("adam"):
            self.optimizer =  torch.optim.Adam(
                self.model.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.get("weight_decay", 0.0),
            )

    def _base_model_dir(self) -> Path:
        output_cfg = getattr(self.config, "output", None)
        if output_cfg is not None and "base_model_path" in output_cfg:
            return Path(str(output_cfg.base_model_path))
        try:
            from hydra.core.hydra_config import HydraConfig
            project_root = Path(HydraConfig.get().runtime.cwd)
        except Exception:
            project_root = Path(os.getcwd())
        dataset_name = str(getattr(getattr(self.config, "dataset", {}), "name", "default"))
        method_name = str(getattr(getattr(self.config, "method", {}), "name", "default"))
        return project_root / "models" / dataset_name / method_name

    def save_pretrained_model(self, pretrained):
        print(pretrained)
        torch.save(self.model.state_dict(), pretrained)
        print("--- Model saved ---")

    def load_pretrained_model(self, pretrained):
        self.model.load_state_dict(torch.load(pretrained, weights_only=True, map_location=self.device))

    def build_base_model(self, retrain=False, **kwargs):
        if retrain:
            assert kwargs['train_loader'] is not None and kwargs['val_loader'] is not None, "Train and validation loaders must exist in order to re-train the model."
            print('going for training')
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
                self.train_base_model(kwargs['train_loader'], kwargs['val_loader'], weights)
            else:
                self.train_base_model(kwargs['train_loader'], kwargs['val_loader'])
            print('train is done')
            base_model_dir = self._base_model_dir()
            base_model_dir.mkdir(parents=True, exist_ok=True)
            if 'model_name' in kwargs:
                model_name = kwargs['model_name']
            else:
                model_name = f'base_model_{self.config.model.name}.pt'
            print(self.model)
            self.save_pretrained_model(str(base_model_dir / model_name))
        else:
            assert kwargs['pretrained'] is not None, "Pretrained checkpoint cannot be None to use a pretrained model."
            self.load_pretrained_model(kwargs['pretrained'])

    @property
    def is_multilabel(self) -> bool:
        return bool(self.config.dataset.get('multilabel', False))

    def train_base_model(self, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, loss_weight=None):
        """Train the model using standard supervised learning."""
        optimizer = self.optimizer
        multilabel = self.is_multilabel
        if multilabel:
            criterion = nn.BCEWithLogitsLoss(pos_weight=loss_weight)
        else:
            criterion = nn.CrossEntropyLoss(loss_weight)

        best_acc = 0.0
        best_previous_loss = float('inf')
        epochs = self.config.optimizer.get('epochs', 10)
        print("Any trainable params:", any(p.requires_grad for p in self.model.parameters()))


        patience = self.config.optimizer.get('early_stopping_patience', None)
        min_delta = self.config.optimizer.get('early_stopping_min_delta', 0.0)
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            total_correct = 0
            total_labels = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()

                train_loss += loss.item() * targets.size(0)
                if multilabel:
                    preds = (outputs > 0).float()
                    total_correct += (preds == targets).sum().item()
                    total_labels += targets.numel()
                else:
                    total_correct += (outputs.argmax(1) == targets).sum().item()
                    total_labels += targets.size(0)


            train_loss = train_loss / len(train_loader.dataset)

            # Validation
            self.model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)

                    val_loss += loss.item() * targets.size(0)
                    if multilabel:
                        preds = (outputs > 0).float()
                        correct += (preds == targets).sum().item()
                        total += targets.numel()
                    else:
                        correct += outputs.max(1)[1].eq(targets).sum().item()
                        total += targets.size(0)

            val_loss = val_loss / len(val_loader.dataset)
            val_acc = correct / total

            print(f'Epoch {epoch+1}/{epochs} - Train: Loss: {train_loss:.4f}, Accuracy: {total_correct / total_labels:.4f}\nValidation: Loss {val_loss:.4f}, Accuracy: {val_acc:.4f}')


            # inside epoch loop, after computing val_loss:
            if patience is not None:
                if best_previous_loss - val_loss > min_delta:
                    best_previous_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

    def inference(self, loader: torch.utils.data.DataLoader):
        """Run deterministic inference and return class probabilities and labels."""
        self.model.eval()
        predictions = []
        labels = []

        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                if self.is_multilabel:
                    probs = torch.sigmoid(outputs)
                else:
                    probs = F.softmax(outputs, dim=1)
                predictions.append(probs)
                labels.append(targets)

        predictions = torch.cat(predictions, dim=0)
        labels = torch.cat(labels, dim=0)
        return predictions, labels

    def measure_uncertainty(self, loader: torch.utils.data.DataLoader):
        """Measure uncertainty. Must be implemented by child classes.

        Returns:
            Dictionary containing uncertainty measures:
                - total_uncertainty
                - aleatoric_uncertainty (data uncertainty)
                - epistemic_uncertainty (model uncertainty)
                - out_of_distribution (OOD score)
        """
        predictions, ground_truth = self.inference(loader)    # shape [T, B, C]
        print(predictions.shape)
        aleatoric_uncertainty = -torch.mean(torch.sum(predictions * torch.log(predictions+self.eps), dim=2), dim=0)

        p_mean = torch.mean(predictions, dim=0)
        total_uncertainty = -torch.sum(p_mean * torch.log(p_mean+self.eps), dim=1)

        epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty

        mi = (predictions * (torch.log(predictions + self.eps) - torch.log(p_mean + self.eps))).sum(dim=2).mean(dim=0)  # [N]

        var_epistemic = predictions.var(dim=0).sum(dim=-1)  # [B, C] --> [B]
        var_aleatoric = (predictions * (1 - predictions)).mean(dim=0).sum(dim=-1)
        var_total = var_epistemic + var_aleatoric

        return {
            "predictions": p_mean,
            "predicted_labels": p_mean.argmax(dim=-1),
            "ground_truth": ground_truth,
            "total_uncertainty": total_uncertainty,
            "aleatoric_uncertainty": aleatoric_uncertainty,
            "epistemic_uncertainty": epistemic_uncertainty,
            "mutual_information": mi,
            "variance_epistemic_uncertainty": var_epistemic,
            "variance_aleatoric_uncertainty": var_aleatoric,
            "variance_total_uncertainty": var_total,
        }

    def train_model(self, train_loader, val_loader, **kwargs):
        """Wrapper for compatibility."""
        self.build_base_model(retrain=True, train_loader=train_loader, val_loader=val_loader)

    def save_model(self, path: str) -> None:
        """Save model to disk."""
        self.save_pretrained_model(path)

    def load_model(self, path: str, train_loader=None, val_loader=None) -> None:
        """Load model from disk.
        
        Args:
            path: Path to model checkpoint
            train_loader: Optional training loader for methods that need to rebuild components
            val_loader: Optional validation loader for methods that need to rebuild components
        """
        self.load_pretrained_model(path)


    def evaluate_method(self, predictions, targets):
        nll = get_NLL_score(predictions, targets)
        accuracy = get_acc_score(predictions, targets)
        auc = get_auc_score(predictions, targets)

        return {
            "nll": nll.detach().cpu().numpy(),
            "accuracy": accuracy.detach().cpu().numpy(),
            "auc": auc
        }
