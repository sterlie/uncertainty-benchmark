import torch
import os
import pickle
from tqdm import tqdm
import numpy as np
from laplace import Laplace

from src.methods.method_factory import register_method
from src.methods.method import Method



#@register_method("influence_function")
class InfluenceFunction(Method):
    def __init__(self, config):
        super(InfluenceFunction, self).__init__(config)

#    def build_method(self, rebuild=False, **kwargs):
#        if not rebuild:
#            self.model.train()
#            self.laplace = Laplace(self.model, "classification",
#                                   self.config.method.subset_of_weights,
#                                   self.config.method.hessian_structure)
#
#            self.laplace.load_state_dict(
#                torch.load(os.path.join(self.config.method.output.path, 'model', 'laplace.pt'), map_location='cpu'))
#            return
#        self.train_uncertainty_method(kwargs['train_dl'], kwargs['valid_dl'])
#        output_save_dir = os.path.join(self.config.output.path, 'model')
#        os.makedirs(output_save_dir, exist_ok=True)
#        with open(os.path.join(output_save_dir, 'laplace.pkl'), 'wb') as f:
#            pickle.dump(self.laplace, f)

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

    def measure_uncertainty(self, loader):
        """
        Computes g^T H^{-1} g using Laplace approximation.
        """
        model = self.laplace.model
        model.eval()

        uncertainty_scores = []
        for x, y in tqdm(loader):
            # Step 1: get gradient w.r.t parameters
            model.zero_grad()
            out = model(x)
            loss = torch.nn.functional.cross_entropy(out, y)
            grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)
            g = torch.cat([p.reshape(-1) for p in grads])

            # 2. extract Hessian diagonal
            H_diag = self.laplace.H  # tensor of shape (num_params,)

            # 3. compute IHVP (closed form)
            ihvp = g / (H_diag + self.laplace.prior_precision)

            # # Step 2: use Laplace to compute inverse-Hessian-vector product
            # ihvp = self.laplace.solve(g)

            # Step 3: compute quadratic form
            uncertainty = torch.dot(g, ihvp).item()
            uncertainty_scores.append(uncertainty)

        uncertainty_scores = np.array(uncertainty_scores)
        return {
            "total_uncertainty": uncertainty_scores,
            "epistemic_uncertainty": uncertainty_scores,
            "aleatoric_uncertainty": 0,
            "out_of_distribution": 0
        }

    def train_model(self, train_loader, val_loader, **kwargs):
        """Train the base model and then train the uncertainty method (Gaussian models)."""
        # Train the base model
        self.train_base_model(train_loader, val_loader)
        # Train method specific model 
        self.train_uncertainty_method(train_loader, val_loader)


    def save_model(self, path: str) -> None:
        """Save the base model and method specific models."""
        # Save the neural network
        self.save_pretrained_model(path)
        
        # Save Gaussian models alongside the network
        if self.laplace is not None:
            laplace_path = path.replace('.pt', '_laplace.pkl')
            with open(laplace_path, 'wb') as f:
                pickle.dump({
                    'laplace': self.laplace,
                }, f)

    def load_model(self, path: str, train_loader=None, val_loader=None) -> None:
        """Load the neural network and Gaussian models.
        
        If method specific pickle doesn't exist and training loaders are provided, train GMMs.
        """
        # Load the neural network
        self.load_pretrained_model(path)
        
        # Try to load Gaussian models
        laplace_path = path.replace('.pt', '_laplace.pkl')
        if os.path.exists(laplace_path):
            with open(laplace_path, 'rb') as f:
                gmm_data = pickle.load(f)
                self.laplace = gmm_data['laplace']

        elif train_loader is not None:
            # If GMM pickle doesn't exist but training loader is available, train GMMs
            print(f"laplace not found at {laplace_path}. Training laplace from training data...")
            self.train_uncertainty_method(train_loader, val_loader)
            # Save the trained models so we don't retrain next time
            self.save_model(path)
        else:
            print(f"Warning: laplace models not found at {laplace_path} and no training loader provided.")
