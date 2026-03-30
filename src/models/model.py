import torch.nn as nn


class Model(nn.Module):
    """Base model class for all architectures in this repository."""

    def __init__(self, config):
        super().__init__()
        self.config = config
