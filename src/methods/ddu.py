from typing import Callable

import numpy as np
import os
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


def entropy(logits, eps=1e-6):
    p = F.softmax(logits, dim=1)
    log_p = F.log_softmax(logits, dim=1)
    p_log_p = p.double() * torch.clamp(log_p, min=np.log(eps)).double()
    entropy = -torch.sum(p_log_p, dim=1)
    return entropy.float()


def logsumexp(logits):
    if logits.ndim == 3:
        score = torch.logsumexp(logits, dim=2, keepdim=False)
        # score = score.mean(dim=1)
    elif logits.ndim == 2:
        score = torch.logsumexp(logits, dim=1, keepdim=False)
    return score


def centered_cov_torch(x):
    n = x.shape[0]
    res = 1 / (n - 1) * x.t().mm(x)
    return res


def get_embeddings(
    method, net, loader: torch.utils.data.DataLoader, handle: Callable, num_dim: int, dtype, device, storage_device,
):
    num_samples = len(loader.dataset)
    embeddings = torch.empty((num_samples, num_dim), dtype=dtype, device=storage_device)

    labels = torch.empty(num_samples, dtype=torch.int, device=storage_device)

    with torch.no_grad():
        start = 0
        for data, label in tqdm(loader):
            data = data.to(device)
            label = label.to(device)

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
        classwise_mean_features = torch.stack([torch.mean(embeddings[labels == c], dim=0) for c in range(num_classes)])
        classwise_cov_features = torch.stack(
            [centered_cov_torch(embeddings[labels == c] - classwise_mean_features[c]) for c in range(num_classes)]
        )
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
        print("acc", (results['predictions'].argmax(dim=-1) == results['ground_truth']).sum(), len(results['ground_truth']))
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
