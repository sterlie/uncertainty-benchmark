import os
from laplace.laplace import Laplace
import torch
import numpy as np
from tqdm import tqdm
import pickle

from src.methods import register_method
from src.methods.method import Method

@register_method("laplace_approximation")
class LaplaceApproximation(Method):
    def __init__(self, config):
        super(LaplaceApproximation, self).__init__(config)

    def build_method(self, rebuild=False, **kwargs):
        if not rebuild:
            self.model.train()
            self.laplace = Laplace(self.model,  "classification",
                               self.config.method.subset_of_weights,
                               self.config.method.hessian_structure)

            model_name =  'model.pt'
            if 'pretrained' in kwargs:
                model_name = os.path.basename(kwargs['pretrained'])
            self.laplace.model.eval()
            self.laplace.load_state_dict(torch.load(os.path.join(self.config.output.path, self.config.method.name, 'model', model_name), map_location=self.device))
            return
        self.train_method(kwargs['train_loader'], kwargs['valid_loader'])
        output_save_dir = os.path.join(self.config.output.path, self.config.method.name, 'model')
        os.makedirs(output_save_dir, exist_ok=True)
        model = self.train_uncertainty_method(kwargs['train_loader'], kwargs['valid_loader'])
        if 'model_name' in kwargs:
            model_name = kwargs['model_name']
        else:
            model_name = f'model.pt'
        torch.save(self.laplace.state_dict(), os.path.join(output_save_dir, model_name))

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
