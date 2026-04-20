import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.methods.method import Method
from src.methods.method_factory import register_method


@register_method("mc_dropout")
class MCDropout(Method):
    """Monte Carlo Dropout for uncertainty quantification.

    This method uses dropout at test time to approximate Bayesian inference.
    Multiple forward passes with different dropout masks provide uncertainty estimates.
    """

    def __init__(self, config):
        super(MCDropout, self).__init__(config)
        self.sample_size = config.method.get('sample_size', 100)
        self.ood_threshold = config.method.get('ood_threshold', 1.0)
        self.misclassify_threshold = config.method.get('misclassify_threshold', self.ood_threshold)

        # Ensure model has dropout layers
        self._enable_dropout()

    def _enable_dropout(self):
        """Enable dropout layers during inference."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
        return

    def measure_uncertainty(self, loader: torch.utils.data.DataLoader):
        results = dict()
        for inputs, targets in loader:
            result = self.do_measure_uncertainty(inputs, targets)
            for key, value in result.items():
                results[key] = torch.cat([results[key], value]) if key in results else value
        return results

    def do_measure_uncertainty(self, inputs: torch.Tensor, targets: torch.Tensor):
        """Measure uncertainty using MC Dropout.

        Args:
            inputs: Input tensor
            targets: Target tensor

        Returns:
            Dictionary with uncertainty measures
        """
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        self.model.eval()
        self._enable_dropout()
        # Keep dropout enabled for MC sampling

        predictions = []
        logit_predictions = []
        with torch.no_grad():
            for _ in range(self.sample_size):
                outputs = self.model(inputs)
                logit_predictions.append(outputs)
                if self.is_multilabel:
                    probs = torch.sigmoid(outputs)
                else:
                    probs = F.softmax(outputs, dim=1)
                predictions.append(probs)

        predictions = torch.stack(predictions)
        # Shape: (sample_size, batch_size, num_classes)

        # Mean prediction
        mean_pred = predictions.mean(dim=0)

        # Total uncertainty (entropy of mean prediction)
        total_uncertainty = -torch.sum(mean_pred * torch.log(mean_pred + self.eps), dim=1)

        # Aleatoric uncertainty (expected entropy)
        entropies = -torch.sum(predictions * torch.log(predictions + self.eps), dim=2)
        aleatoric_uncertainty = entropies.mean(dim=0)

        # Epistemic uncertainty (mutual information)
        epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty

        var_epistemic = predictions.var(dim=0).sum(dim=-1)  # [B, C] --> [B]
        var_aleatoric = (predictions * (1 - predictions)).mean(dim=0).sum(dim=-1)
        var_total = var_epistemic + var_aleatoric

        # OOD score (variance in predictions)
        variance = predictions.var(dim=0)
        variance = variance[torch.arange(variance.shape[0]), mean_pred.argmax(1).type(torch.int)]
        # OOD detection
        ood_score = total_uncertainty
        misclassify_score = total_uncertainty

        mi = (predictions * (torch.log(predictions + self.eps) - torch.log(mean_pred + self.eps))).sum(dim=2).mean(dim=0)  # [N]

        return {
            "predictions": mean_pred,
            "predicted_labels": mean_pred.argmax(dim=-1),
            "ground_truth": targets,
            "total_uncertainty": total_uncertainty,
            "aleatoric_uncertainty": aleatoric_uncertainty,
            "epistemic_uncertainty": epistemic_uncertainty,
            "mutual_information": mi,
            "variance_epistemic_uncertainty": var_epistemic,
            "variance_aleatoric_uncertainty": var_aleatoric,
            "variance_total_uncertainty": var_total
        }