import torch

import torch.nn as nn

from torchvision.transforms.v2 import JPEG as JPEGTransform


class JPEG(nn.Module):
    def __init__(self, config):
        super(JPEG, self).__init__()
        self.config = config

    def forward(self, x):
        """Forward function.
        Args:
            x: Input feature, tensor size (B, C, T, H, W).
        """
        x = x.permute(0, 2, 1, 3, 4).to(torch.uint8)  # (B, T, C, H, W)
        x = JPEGTransform(quality=self.config.model.encoder.quality)(x)
        return x

    def get_temporal_dim(self, window_size):
        return self.config.data.window_size

    def get_encoding_dim(self):
        return 100
