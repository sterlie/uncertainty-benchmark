from src.methods.method import Method
from laplace.laplace import Laplace
import torch
import os
import pickle
from tqdm import tqdm
import numpy as np

class InfluenceFunction(Method):
    def __init__(self, config):
        super(InfluenceFunction, self).__init__(config)

    def build_method(self, rebuild=False, **kwargs):
        if not rebuild:
            self.model.train()
            self.laplace = Laplace(self.model, "classification",
                                   self.config.method.subset_of_weights,
                                   self.config.method.hessian_structure)

            self.laplace.load_state_dict(
                torch.load(os.path.join(self.config.method.output.path, 'model', 'laplace.pt'), map_location='cpu'))
            return
        self.train_method(kwargs['train_dl'], kwargs['valid_dl'])
        output_save_dir = os.path.join(self.config.output.path, 'model')
        os.makedirs(output_save_dir, exist_ok=True)
        with open(os.path.join(output_save_dir, 'laplace.pkl'), 'wb') as f:
            pickle.dump(self.laplace, f)

    def train_method(self, train_loader, val_loader):
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

