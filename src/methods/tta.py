import torch
import torch.nn.functional as F
from torchvision.transforms import *
from torchvision.transforms.v2 import *
from tqdm import tqdm
from torch import nn

from src.methods import register_method
from src.methods.method import Method

def build_transform(cfg):
    """Recursively build transforms (supports Compose, RandomChoice, etc.)."""
    class_path = cfg['_type']
    cls = globals()[class_path]

    # Handle nested transforms
    if 'transforms' in cfg:
        sub_transforms = [build_transform(t) for t in cfg['transforms']]
        kwargs = {k: v for k, v in cfg.items() if k not in ['_type', 'transforms']}
        return cls(sub_transforms, **kwargs)

    # Simple transform
    kwargs = {k: v for k, v in cfg.items() if k != '_type'}
    return cls(**kwargs)

@register_method('tta')
class TTA(Method):
    def __init__(self, config):
        super(TTA, self).__init__(config)
        self.default_augmentation = build_transform(config.method.augmentation)
        print(self.default_augmentation)

    def inference(self, loader: torch.utils.data.DataLoader, enable_augmentation=True, enable_dropout=False):
        self.model.eval()
        if enable_dropout:
            for module in self.model.modules():
                if isinstance(module, nn.Dropout):
                    module.train()

        outputs = []
        T = self.config.get('sample_size', 100)
        labels = []
        with torch.no_grad():
            for inputs_, targets_ in tqdm(loader):
                if enable_augmentation:
                    x = torch.stack(
                        [self.default_augmentation(in_) for _ in range(T) for in_ in inputs_],
                        dim=0
                    )
                    # x = x*255.    ## use if images were not normalized
                else:
                    x = torch.cat([inputs_ for _ in range(T)], dim=0)
                x = x.to(self.device)
                output = self.model(x)
                if self.is_multilabel:
                    output = torch.sigmoid(output)
                else:
                    output = F.softmax(output, dim=-1)
                output = output.view(T, len(targets_), -1)
                outputs.append(output)
                labels.append(targets_)

        predictions = torch.cat(outputs, dim=1)
        labels = torch.cat(labels, dim=0)
        return predictions, labels

    def measure_uncertainty(self, loader: torch.utils.data.DataLoader):
        """Measure uncertainty. Must be implemented by child classes.

        Returns:
            Dictionary containing uncertainty measures:
                - total_uncertainty
                - aleatoric_uncertainty (data uncertainty)
                - epistemic_uncertainty (model uncertainty)
                - out_of_distribution (OOD score)
        """
        predictions, labels = self.inference(loader, enable_augmentation=True)

        p_mean = torch.mean(predictions, dim=0)
        aleatoric_uncertainty = -torch.sum(p_mean * torch.log(p_mean + self.eps), dim=1)

        predictions_, _ = self.inference(loader, enable_augmentation=False, enable_dropout=True)

        mean_pred = predictions_.mean(dim=0)
        epistemic_uncertainty = -torch.sum(mean_pred * torch.log(mean_pred + self.eps), dim=1)

        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty

        var_epistemic = predictions.var(dim=0).sum(dim=-1)  # [B, C] --> [B]
        var_aleatoric = (predictions * (1 - predictions)).mean(dim=0).sum(dim=-1)
        var_total = var_epistemic + var_aleatoric

        return {
            "predictions": p_mean,
            "predicted_labels": predictions.argmax(dim=-1).mode(dim=0).values,
            "ground_truth": labels,
            "total_uncertainty": aleatoric_uncertainty,
            "aleatoric_uncertainty": aleatoric_uncertainty,
            "epistemic_uncertainty": epistemic_uncertainty,
            "mutual_information": torch.zeros(total_uncertainty.size(0)),
            "variance_epistemic_uncertainty": var_epistemic,
            "variance_aleatoric_uncertainty": var_aleatoric,
            "variance_total_uncertainty": var_total,
            "out_of_distribution": torch.zeros(total_uncertainty.size(0)),
        }