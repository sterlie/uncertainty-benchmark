import os
from laplace.laplace import Laplace
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from src.methods import register_method
from src.methods.method import Method


class RegressionDataset(torch.utils.data.Dataset):
    """Maps binary multi-label targets to (+/-)scale for Laplace regression likelihood."""

    def __init__(self, dataset, scale: float = 10.0):
        self.dataset = dataset
        self.scale = scale

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        items = self.dataset[idx]
        x, y = items[:2]
        y = torch.where(
            y >= 0.5,
            self.scale * torch.ones_like(y, dtype=torch.float32),
            -self.scale * torch.ones_like(y, dtype=torch.float32),
        )
        if len(items) > 3:
            return x, y, items[2], items[3]
        if len(items) > 2:
            return x, y, items[2]
        return x, y

@register_method("laplace_approximation")
class LaplaceApproximation(Method):
    def __init__(self, config):
        super(LaplaceApproximation, self).__init__(config)
        self.laplace = None
        self.la_batch_size = config.method.get('la_batch_size', 4)
        self.train_size = config.method.get('train_size', 1000)
        self.val_size = config.method.get('val_size', 100)

    def train_model(self, train_loader, val_loader, **kwargs):
        """Train the base model and then fit Laplace approximation."""
        # Train the base model
        self.train_base_model(train_loader, val_loader)
        # Train the Laplace approximation
        self.train_uncertainty_method(train_loader, val_loader)

    def load_model(self, path: str, train_loader=None, val_loader=None) -> None:
        """Load the base model and restore Laplace approximation."""
        self.load_pretrained_model(path)

        # Try to load Laplace state_dict
        laplace_path = path.replace('.pt', '_laplace.pt')
        if os.path.exists(laplace_path):
            self.model.eval()
            likelihood = "regression" if self.is_multilabel else "classification"
            self.laplace = Laplace(
                self.model,
                likelihood,
                self.config.method.subset_of_weights,
                self.config.method.hessian_structure,
            )
            checkpoint = torch.load(laplace_path, weights_only=True, map_location=self.device)
            self.laplace.load_state_dict(checkpoint["laplace_state_dict"])
            self.laplace.prior_precision = checkpoint["laplace_config"]["prior_precision"]
            self.laplace.temperature = checkpoint["laplace_config"]["temperature"]
            return

        raise FileNotFoundError(
            f"Laplace state not found at {laplace_path}. Delete the base model and retrain."
        )

    def save_model(self, path: str) -> None:
        """Save the model and Laplace approximation state."""
        # Save the neural network
        self.save_pretrained_model(path)

        # Save Laplace approximation state_dict alongside the network
        if self.laplace is not None:
            laplace_path = path.replace('.pt', '_laplace.pt')
            checkpoint = {
                "laplace_state_dict": self.laplace.state_dict(),
                "laplace_config": {
                    "prior_precision": self.laplace.prior_precision,
                    "temperature": self.laplace.temperature,
                },
            }
            torch.save(checkpoint, laplace_path)


    def train_uncertainty_method(self, train_loader, val_loader):
        self.model.eval()
        likelihood = "regression" if self.is_multilabel else "classification"

        train_size = min(self.train_size, len(train_loader.dataset))
        val_size = min(self.val_size, len(val_loader.dataset))

        if self.is_multilabel:
            train_dataset = RegressionDataset(train_loader.dataset)
            val_dataset = RegressionDataset(val_loader.dataset)
        else:
            train_dataset = train_loader.dataset
            val_dataset = val_loader.dataset

        la_train_loader = DataLoader(
            Subset(train_dataset, range(train_size)),
            batch_size=self.la_batch_size, shuffle=True,
        )
        la_val_loader = DataLoader(
            Subset(val_dataset, range(val_size)),
            batch_size=self.la_batch_size, shuffle=False,
        )

        self.laplace = Laplace(
            self.model, likelihood,
            self.config.method.subset_of_weights,
            self.config.method.hessian_structure,
        )
        self.laplace.fit(la_train_loader)
        self.laplace.optimize_prior_precision(
            method=self.config.method.get('optimization_method', 'gridsearch'),
            pred_type=self.config.method.get('pred_type', 'glm'),
            link_approx=self.config.method.get('link_approx', 'mc'),
            val_loader=la_val_loader,
        )

    def inference(self, loader):
        self.model.eval()
        self.laplace.model.eval()
        predictions = []
        labels = []
        for x_test, y_test in tqdm(loader):
            try:
                # User-specified predictive approx.Laplace Redux – Effortless Bayesian Deep Learning
                pred = self.laplace.predictive_samples(x_test.to(self.device), n_samples=self.config.method.num_samples)
                # regression likelihood returns raw Gaussian samples in logit space;
                # apply sigmoid to get probabilities for multilabel uncertainty.
                if self.is_multilabel:
                    pred = torch.sigmoid(pred)
                predictions.append(pred)
                labels.append(y_test.to(self.device))
            except Exception as e:
                print(e)
                # pass

        predictions = torch.cat(predictions, dim=1)
        labels = torch.cat(labels, dim=0)
        return predictions, labels
