import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.methods.method import Method
from src.methods.method_factory import register_method
from src.methods.utils import bernoulli_entropy


@register_method("entropy")
class Entropy(Method):
    """Entropy for uncertainty quantification.

    This method uses entropy to measure uncertainty.
    """

    def __init__(self, config):
        super(Entropy, self).__init__(config)
        self.uncertainty_per_class = bool(
            config.method.get('uncertainty_per_class', config.dataset.get('uncertainty_per_class', False))
        )
        self.ood_threshold = config.method.get('ood_threshold', 1.0)
        self.misclassify_threshold = config.method.get('misclassify_threshold', self.ood_threshold)

    def measure_uncertainty(self, loader):
        """Measure uncertainty using Entropy.

        Args:
            inputs: Input tensor
            targets: Target tensor

        Returns:
            Dictionary with uncertainty measures
        """
        predictions, ground_truth = self.inference(loader)
        print(predictions.shape)

        predictions = predictions.to(self.device)
        ground_truth = ground_truth.to(self.device)

        if self.is_multilabel:
            total_uncertainty = bernoulli_entropy(predictions, self.eps)
            if not self.uncertainty_per_class:
                total_uncertainty = total_uncertainty.mean(dim=1)
        else:
            total_uncertainty = -torch.sum(predictions * torch.log(predictions + self.eps), dim=1)

        aleatoric_uncertainty = total_uncertainty
        epistemic_uncertainty = total_uncertainty
        zero_uncertainty = torch.zeros_like(total_uncertainty)

        return {
            "predictions": predictions,
            "predicted_labels": (predictions > 0.5).long() if self.is_multilabel else predictions.argmax(dim=-1),
            "ground_truth": ground_truth,
            "total_uncertainty": total_uncertainty,
            "aleatoric_uncertainty": aleatoric_uncertainty,
            "epistemic_uncertainty": epistemic_uncertainty,
            "mutual_information": zero_uncertainty,
            "variance_epistemic_uncertainty": zero_uncertainty,
            "variance_aleatoric_uncertainty": zero_uncertainty,
            "variance_total_uncertainty": zero_uncertainty,
            "out_of_distribution": total_uncertainty,
            "misclassification": total_uncertainty,
            "ambiguous": aleatoric_uncertainty,
        }
