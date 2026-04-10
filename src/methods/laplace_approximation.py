import os
from laplace.laplace import Laplace
import torch
import numpy as np
from tqdm import tqdm

from src.methods import register_method
from src.methods.method import Method

@register_method("laplace_approximation")
class LaplaceApproximation(Method):
    def __init__(self, config):
        super(LaplaceApproximation, self).__init__(config)
        self.laplace = None

    def train_model(self, train_loader, val_loader, **kwargs):
        """Train the base model and then fit Laplace approximation."""
        # Train the base neural network
        self.train_base_model(train_loader, val_loader)
        # Train the Laplace approximation
        self.train_uncertainty_method(train_loader, val_loader)

    def load_model(self, path: str, train_loader=None, val_loader=None) -> None:
        """Load the base model and restore Laplace approximation."""
        self.load_pretrained_model(path)

        # Try to load Laplace state_dict
        laplace_path = path.replace('.pt', '_laplace.pt')
        if os.path.exists(laplace_path):

            # eval mode before loading state_dict 
            self.model.eval()
            self.laplace = Laplace(
                self.model,
                "classification",
                self.config.method.subset_of_weights,
                self.config.method.hessian_structure,
            )
            laplace_state = torch.load(laplace_path, map_location=self.device)
            self.laplace.load_state_dict(laplace_state)
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
            torch.save(self.laplace.state_dict(), laplace_path)

#    def build_method(self, rebuild=False, **kwargs):
#        if not rebuild:
#            self.model.eval()
#            self.laplace = Laplace(self.model,  "classification",
#                               self.config.method.subset_of_weights,
#                               self.config.method.hessian_structure)
#
#            model_name =  'model.pt'
#            if 'pretrained' in kwargs:
#                model_name = os.path.basename(kwargs['pretrained'])
#            self.laplace.model.eval()
#            self.laplace.load_state_dict(torch.load(os.path.join(self.config.output.path, self.config.method.name, 'model', model_name), map_location=self.device))
#            return
#        self.train_method(kwargs['train_loader'], kwargs['valid_loader'])
#        output_save_dir = os.path.join(self.config.output.path, self.config.method.name, 'model')
#        os.makedirs(output_save_dir, exist_ok=True)
#        model = self.train_uncertainty_method(kwargs['train_loader'], kwargs['valid_loader'])
#        if 'model_name' in kwargs:
#            model_name = kwargs['model_name']
#        else:
#            model_name = f'model.pt'
#        torch.save(self.laplace.state_dict(), os.path.join(output_save_dir, model_name))

    def train_uncertainty_method(self, train_loader, val_loader):
        self.model.eval()
        self.laplace = Laplace(self.model, "classification",
                               self.config.method.subset_of_weights,
                               self.config.method.hessian_structure)

        self.laplace.fit(train_loader)
        self.laplace.optimize_prior_precision(
            method="gridsearch",
            pred_type="glm",
            link_approx="mc",
            val_loader=val_loader
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
                predictions.append(pred)
                labels.append(y_test.to(self.device))
            except Exception as e:
                print(e)
                # pass

        predictions = torch.cat(predictions, dim=1)
        labels = torch.cat(labels, dim=0)
        return predictions, labels
