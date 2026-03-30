import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.methods.method import Method
from src.methods.method_factory import register_method


@register_method("entropy")
class Entropy(Method):
    """Entropy for uncertainty quantification.

    This method uses entropy to measure uncertainty.
    """

    def __init__(self, config):
        super(Entropy, self).__init__(config)
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

        total_uncertainty = -torch.sum(predictions * torch.log(predictions + self.eps), dim=1)

        return {
            "predictions": predictions,
            "predicted_labels": predictions.argmax(dim=-1),
            "ground_truth": ground_truth,
            "total_uncertainty": total_uncertainty,
            "aleatoric_uncertainty": torch.zeros(total_uncertainty.size(0)),
            "epistemic_uncertainty": torch.zeros(total_uncertainty.size(0)),
            "mutual_information": torch.zeros(total_uncertainty.size(0)),
            "variance_epistemic_uncertainty": torch.zeros(total_uncertainty.size(0)),
            "variance_aleatoric_uncertainty": torch.zeros(total_uncertainty.size(0)),
            "variance_total_uncertainty": torch.zeros(total_uncertainty.size(0)),        }
