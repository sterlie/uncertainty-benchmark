import torch.nn as nn

from .model import Model
from .model_factory import register_model


@register_model("mlp")
class MLP(Model):
    """Multi-layer perceptron with configurable depth/width."""

    def __init__(self, config):
        super().__init__(config)

        input_dim = config.dataset.get("input_dim", 784)
        output_dim = config.dataset.get("num_classes", 10)

        hidden_dim = config.model.get("hidden_dim", 256)
        num_layers = config.model.get("num_layers", 3)
        dropout = config.model.get("dropout", 0.0)
        activation = config.model.get("activation", "ReLU")
        normalization = config.model.get('normalization', None)

        layers = []
        in_features = input_dim

        for _ in range(max(num_layers - 1, 1)):
            layers.append(nn.Linear(in_features, hidden_dim))
            norm_layer = self._get_normalization(normalization, hidden_dim)
            if norm_layer is not None:
                layers.append(norm_layer)
            layers.append(self._get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden_dim

        layers.append(nn.Linear(in_features, output_dim))
        self.network = nn.Sequential(*layers)

    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        activations = {
            "ReLU": nn.ReLU(),
            "LeakyReLU": nn.LeakyReLU(),
            "ELU": nn.ELU(),
            "GELU": nn.GELU(),
            "Tanh": nn.Tanh(),
            "Sigmoid": nn.Sigmoid(),
        }
        return activations.get(name, nn.ReLU())

    @staticmethod
    def _get_normalization(name: str | None, features: int) -> nn.Module | None:
        if name == "BatchNorm":
            return nn.BatchNorm1d(features)
        if name == "LayerNorm":
            return nn.LayerNorm(features)
        return None

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)
