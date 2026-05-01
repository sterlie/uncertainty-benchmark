from typing import Callable

import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn.utils.parametrizations import spectral_norm as SN

from src.methods.method import Method
from src.methods.method_factory import register_method

DOUBLE_INFO = torch.finfo(torch.double)
JITTERS = [0, DOUBLE_INFO.tiny] + [10 ** exp for exp in range(-308, 0, 1)]
# 2) Which layers can receive SpectralNorm?
_SN_ELIGIBLE = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)
REMOVE_SN = lambda m: nn.utils.parametrize.remove_parametrizations(m, "weight", leave_parametrized=True)



def entropy(logits, multi_label=False, reduction=True, eps=1e-6):
    if multi_label:
        probs = torch.sigmoid(logits).clamp(min=eps, max=1 - eps)
        entropy = -(probs * torch.log(probs) + (1 - probs) * torch.log(1 - probs))
        entropy = entropy.mean(dim=1) if reduction else entropy
    else:
        p = F.softmax(logits, dim=1)
        log_p = F.log_softmax(logits, dim=1)
        p_log_p = p.double() * torch.clamp(log_p, min=np.log(eps)).double()
        entropy = -torch.sum(p_log_p, dim=1)
    return entropy.float()


def logsumexp(logits, multi_label=False, reduction=True):
    if multi_label:
        if logits.ndim == 3:
            if logits.shape[2] == 1:
                score = torch.max(torch.logsumexp(logits, dim=2, keepdim=False), torch.logsumexp(-logits, dim=2, keepdim=False))
            else:
                score = torch.max(torch.logsumexp(logits[..., :1], dim=2, keepdim=False), torch.logsumexp(logits[..., 1:], dim=2, keepdim=False))
            score = score.mean(dim=1) if reduction else score
        else:
            score = torch.logsumexp(logits, dim=1, keepdim=False)
            score = score if reduction else score.unsqueeze(1).repeat(1, logits.shape[1])
    else:
        score = torch.logsumexp(logits, dim=1, keepdim=False)
    return score


def centered_cov_torch(x):
    n = x.shape[0]
    if n <= 1:
        # Not enough samples to compute covariance; return identity matrix
        return torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
    res = 1 / (n - 1) * x.t().mm(x)
    return res


def get_embeddings(
    method, net, loader: torch.utils.data.DataLoader, handle: Callable, num_dim: int, dtype, device, storage_device,
):
    num_samples = len(loader.dataset)
    embeddings = torch.empty((num_samples, num_dim), dtype=dtype, device=storage_device)

    # Peek at first batch to determine label shape (single-label 1D vs multi-label 2D)
    first_data, first_label = next(iter(loader))
    first_label = first_label.to(device)
    if first_label.ndim == 2 and first_label.shape[1] > 1:
        # multi-label: keep (N, num_classes)
        labels = torch.empty((num_samples, first_label.shape[1]), dtype=torch.int, device=storage_device)
        multi_label = True
    else:
        labels = torch.empty(num_samples, dtype=torch.int, device=storage_device)
        multi_label = False

    with torch.no_grad():
        start = 0
        for data, label in tqdm(loader):
            data = data.to(device)
            label = label.to(device)

            if multi_label:
                label = label.long()
            else:
                label = label.squeeze(1).long() if label.ndim == 2 else label.long()

            if isinstance(net, nn.DataParallel):
                out = net.module(data)
                out = net.module.features if hasattr(net.module, "features") else method.features
            else:
                out = net(data)
                out = net.features if hasattr(net, "features") else method.features

            end = start + len(data)
            embeddings[start:end].copy_(out, non_blocking=True)
            labels[start:end].copy_(label, non_blocking=True)
            start = end

    return embeddings, labels


def gmm_forward(method, net, gaussians_model, data_B_X):
    if isinstance(net, nn.DataParallel):
        logits_B_Z = net.module(data_B_X)
        features_B_Z = net.module.features if hasattr(net.module, "features") else method.features
    else:
        logits_B_Z = net(data_B_X)
        features_B_Z = net.features if hasattr(net, "features") else method.features

    # logits_B_Z = torch.sigmoid(logits_B_Z) if multi_label else F.softmax(logits_B_Z, dim=-1)
    if isinstance(gaussians_model, list):
        log_probs_B_Y = []
        for gmm in gaussians_model:
            gmm_out = gmm.log_prob(features_B_Z[:, None, :])
            log_probs_B_Y.append(gmm_out.unsqueeze(1))
        log_probs_B_Y = torch.cat(log_probs_B_Y, dim=1)
        # print(log_probs_B_Y.shape, torch.sigmoid(log_probs_B_Y))
    else:
        log_probs_B_Y = gaussians_model.log_prob(features_B_Z[:, None, :])
    return logits_B_Z, log_probs_B_Y, features_B_Z


def gmm_evaluate(net, gaussians_model, loader, device, num_classes, storage_device):
    num_samples = len(loader.dataset)
    logits_N_C = torch.empty((num_samples, num_classes), dtype=torch.float, device=storage_device)
    labels_N = torch.empty(num_samples, dtype=torch.int, device=storage_device)

    with torch.no_grad():
        start = 0
        for data, label in tqdm(loader):
            data = data.to(device)
            label = label.to(device)

            label = label.squeeze(1).long() if label.ndim == 2 else label.long()

            _, logit_B_C, _ = gmm_forward(net, gaussians_model, data)

            end = start + len(data)
            logits_N_C[start:end].copy_(logit_B_C, non_blocking=True)
            labels_N[start:end].copy_(label, non_blocking=True)
            start = end

    return logits_N_C, labels_N


def gmm_get_logits(gmm, embeddings):
    log_probs_B_Y = gmm.log_prob(embeddings[:, None, :])
    return log_probs_B_Y


def gmm_fit(embeddings, labels, num_classes):
    with torch.no_grad():
        if labels.ndim == 2:
            # multi-label: for class c select samples where label[:, c] == 1
            def _mask(c):
                return labels[:, c].bool()
        else:
            def _mask(c):
                return labels == c

        num_dim = embeddings.shape[1]
        classwise_mean_features = []
        classwise_cov_features = []
        for c in range(num_classes):
            class_embs = embeddings[_mask(c)]
            if class_embs.shape[0] == 0:
                # No samples for this class — use zero mean and identity covariance
                mean = torch.zeros(num_dim, dtype=embeddings.dtype, device=embeddings.device)
                cov = torch.eye(num_dim, dtype=embeddings.dtype, device=embeddings.device)
            else:
                mean = torch.mean(class_embs, dim=0)
                cov = centered_cov_torch(class_embs - mean)
            classwise_mean_features.append(mean)
            classwise_cov_features.append(cov)

        classwise_mean_features = torch.stack(classwise_mean_features)
        classwise_cov_features = torch.stack(classwise_cov_features)
        print(classwise_mean_features.shape, classwise_cov_features.shape)

    with torch.no_grad():
        for jitter_eps in JITTERS:
            try:
                jitter = jitter_eps * torch.eye(
                    classwise_cov_features.shape[1], device=classwise_cov_features.device,
                ).unsqueeze(0)
                gmm = torch.distributions.MultivariateNormal(
                    loc=classwise_mean_features, covariance_matrix=(classwise_cov_features + jitter),
                )
            except RuntimeError as e:
                if "cholesky" in str(e):
                    continue
            except ValueError as e:
                if "parameter covariance_matrix" in str(e) and "invalid values" in str(e):
                    continue
            break

    return gmm, jitter_eps


def apply_sn(
    module: nn.Module,
    sn_power_iters: int = 1,
    sn_eps: float = 1e-12,
) -> nn.Module:
    """
    Recursively:
      - apply spectral norm to Conv/Linear layers (if not already applied)
    """
    for name, child in list(module.named_children()):
        # Recurse first
        apply_sn(child, sn_power_iters, sn_eps)

        # If eligible for SN, apply it (idempotent if already applied)
        if isinstance(child, _SN_ELIGIBLE):
            # With new API this is idempotent; older API will error if applied twice.
            try:
                wrapped = SN(child, n_power_iterations=sn_power_iters, eps=sn_eps)
            except TypeError:
                # older torch.nn.utils.spectral_norm signature
                wrapped = SN(child, n_power_iterations=sn_power_iters)
            setattr(module, name, wrapped)

    return module


def has_spectral_norm(m: nn.Module) -> bool:
    # New API stores parametrization; old API registers buffers like weight_u/weight_v.
    return any("weight_orig" in d for d in m._parameters.keys()) or \
        hasattr(m, "weight_u") or hasattr(m, "parametrizations")


def remove_all_spectral_norms(module: nn.Module):
    for name, child in list(module.named_children()):
        if isinstance(child, _SN_ELIGIBLE):
            try:
                REMOVE_SN(child)
            except Exception:
                pass
        remove_all_spectral_norms(child)
    return module


@register_method("ddu")
class DDU(Method):
    """Denoising Diffusion Uncertainty for uncertainty quantification."""

    def hook(self, module, input, output):
        self.features = output.detach()

    def __init__(self, config):
        super(DDU, self).__init__(config)
        self.classification = self.config.method.get('classification', 'basic')
        self.uncertainty = self.config.method.get('uncertainty', 'logsumexp')
        assert self.classification in ['basic', 'gmm']
        assert self.uncertainty in ['logsumexp', 'entropy']
        self.model_dir = "ddu"
        self.uncertainty = self.uncertainty

        self.named_modules = dict(self.model.named_modules())
        self.key = list(self.named_modules.keys())[-2]
        self.features = None
        apply_sn(self.model)
        self.ood_threshold = config.method.get('ood_threshold', 0.00001)
        self.misclassify_threshold = config.method.get('misclassify_threshold', self.ood_threshold)
        self.embeddings = self.labels = self.gaussian_models = self.jitter_eps = None

    def build_base_model(self, retrain=False, **kwargs):
        pass

    def build_method(self, rebuild=False, **kwargs):
        if not rebuild:
            if 'pretrained' in kwargs:
                pretrained_model = os.path.join(self.config.output.path, self.config.method.name, 'model', os.path.basename(kwargs['pretrained']))
            else:
                pretrained_model = os.path.join(self.config.output.path, self.config.method.name, 'model', 'model.pt')
            if self.load_pretrained_model(pretrained_model):
                self.train_uncertainty_method(kwargs['train_loader'], kwargs['valid_loader'])
                return

        else:
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
                self.train_base_model(kwargs['train_loader'], kwargs['valid_loader'], weights)
            else:
                self.train_base_model(kwargs['train_loader'], kwargs['valid_loader'])
            output_save_dir = os.path.join(self.config.output.path, self.config.method.name, 'model')
            os.makedirs(output_save_dir, exist_ok=True)
            if 'model_name' in kwargs:
                filename = os.path.join(output_save_dir,kwargs['model_name'])
            else:
                filename = os.path.join(output_save_dir, 'model.pt')
            self.save_pretrained_model(filename)
            self.train_uncertainty_method(kwargs['train_loader'], kwargs['valid_loader'])

    def train_uncertainty_method(self, train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader):
        """Train the model using standard supervised learning. Then, train the GM models.

        Args:
            loader: Training data loader
        """
        self.model.eval()

        self.handle = self.named_modules[self.key].register_forward_hook(self.hook)
        self.embeddings, self.labels = get_embeddings(
            self,
            self.model,
            train_loader,
            self.handle,
            num_dim=self.config.model.get('hidden_dim', 512),
            dtype=torch.double,
            device=self.device,
            storage_device=self.device
        )
        self.handle.remove()
        self.gaussian_models, self.jitter_eps = gmm_fit(
            embeddings=self.embeddings,
            labels=self.labels,
            num_classes=self.num_classes
        )

    def run_model(self, inputs: torch.Tensor):
        if self.classification == 'basic':
            return self.model(inputs)

        self.handle = self.named_modules[self.key].register_forward_hook(self.hook)
        logits, logits_feat, features = gmm_forward(self, self.model, self.gaussian_models, inputs)
        self.handle.remove()

        predictions = self.gaussian_models.log_prob(features[:, None, :])
        return logits


    def measure_uncertainty(self, loader: torch.utils.data.DataLoader):
        results = dict()
        gt = []
        for inputs, targets in loader:
            result = self.do_measure_uncertainty(inputs, targets)
            for key, value in result.items():
                results[key] = torch.cat([results[key], value]) if key in results else value
            gt.extend(targets)
        preds = results['predictions']
        gt_tensor = results['ground_truth']
        if gt_tensor.ndim == 2:
            # multi-label: count samples where every label matches
            correct = ((preds > 0.5).long() == gt_tensor.long()).all(dim=-1).sum()
        else:
            correct = (preds.argmax(dim=-1) == gt_tensor).sum()
        print("acc", correct.item(), len(gt_tensor))
        return results

    def do_measure_uncertainty(self, inputs: torch.Tensor, targets: torch.Tensor):
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        self.model.eval()

        self.handle = self.named_modules[self.key].register_forward_hook(self.hook)
        with torch.no_grad():
            logits, logits_feat, features = gmm_forward(self, self.model, self.gaussian_models, inputs)
        self.handle.remove()

        uncertainty_function = (lambda x: entropy(x)) if self.uncertainty == 'entropy' else logsumexp
        aleatoric_uncertainty = uncertainty_function(logits)

        epistemic_uncertainty = uncertainty_function(logits_feat)

        total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty
        if self.uncertainty == "logsumexp":
            total_uncertainty = -total_uncertainty
            aleatoric_uncertainty = -aleatoric_uncertainty
            epistemic_uncertainty = -epistemic_uncertainty

        ood_score = epistemic_uncertainty
        misclassify_score = total_uncertainty
        ambiguous_score = aleatoric_uncertainty
        predictions = F.softmax(logits, dim=-1)


        return {
            "predictions": predictions,
            "predicted_labels": predictions.argmax(dim=-1),
            "ground_truth": targets,
            "total_uncertainty": total_uncertainty,
            "aleatoric_uncertainty": aleatoric_uncertainty,
            "epistemic_uncertainty": epistemic_uncertainty,
            "mutual_information": torch.zeros(total_uncertainty.size(0)),
            "variance_epistemic_uncertainty": torch.zeros(total_uncertainty.size(0)),
            "variance_aleatoric_uncertainty": torch.zeros(total_uncertainty.size(0)),
            "variance_total_uncertainty": torch.zeros(total_uncertainty.size(0)),
        }


    def train_model(self, train_loader, val_loader, **kwargs):
        """Train the base model and then train the uncertainty method (Gaussian models)."""
        # Train the neural network
        self.train_base_model(train_loader, val_loader)
        # Train the Gaussian mixture models
        self.train_uncertainty_method(train_loader, val_loader)

    def save_model(self, path: str) -> None:
        """Save the neural network and Gaussian models."""
        # Save the neural network
        self.save_pretrained_model(path)
        
        # Save Gaussian models alongside the network
        if self.gaussian_models is not None:
            gmm_path = path.replace('.pt', '_gmm.pkl')
            with open(gmm_path, 'wb') as f:
                pickle.dump({
                    'gaussian_models': self.gaussian_models,
                    'jitter_eps': self.jitter_eps,
                }, f)

    def load_model(self, path: str, train_loader=None, val_loader=None) -> None:
        """Load the neural network and Gaussian models.
        
        If GMM pickle doesn't exist and training loaders are provided, train GMMs.
        """
        # Load the neural network
        self.load_pretrained_model(path)
        
        # Try to load Gaussian models
        gmm_path = path.replace('.pt', '_gmm.pkl')
        if os.path.exists(gmm_path):
            with open(gmm_path, 'rb') as f:
                gmm_data = pickle.load(f)
                self.gaussian_models = gmm_data['gaussian_models']
                self.jitter_eps = gmm_data['jitter_eps']
        elif train_loader is not None:
            # If GMM pickle doesn't exist but training loader is available, train GMMs
            print(f"GMM models not found at {gmm_path}. Training GMMs from training data...")
            self.train_uncertainty_method(train_loader, val_loader)
            # Save the trained models so we don't retrain next time
            self.save_model(path)
        else:
            print(f"Warning: GMM models not found at {gmm_path} and no training loader provided.")
