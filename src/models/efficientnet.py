from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch import nn

from .model import Model
from .model_factory import register_model

@register_model("efficientnet")
class EfficientNet(Model):

    def __init__(self, config):
        super(EfficientNet, self).__init__(config)

        output_dim = config.dataset.get('num_classes', 10)  # Default for MNIST
        drop_rate = config.model.get('drop_rate', 0.5)
        hidden_dim = config.model.get('hidden_dim', 128)
        # Backbone
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        # Classification head
        self.fc1 = nn.Linear(1000, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(p=drop_rate)

        self.fc3 = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):
        x = self.backbone(x)

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop1(x)
        x = self.fc3(x)
        return x