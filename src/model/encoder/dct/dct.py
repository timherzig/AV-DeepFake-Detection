import torch

from torchvision.transforms import Grayscale

import torch.nn as nn
import torch_dct as dct


class DCT(nn.Module):
    def __init__(self, config):
        super(DCT, self).__init__()
        self.config = config

    def forward(self, x):
        """Forward function.
        Args:
            x: Input feature, tensor size (B, C, T, H, W).
        """
        x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        x = Grayscale()(x).squeeze()
        # convert to 8x8 blocks
        x = x.unfold(2, 8, 8).unfold(3, 8, 8)  # (B, T, C, 28, 28, 8, 8)
        print(f"unfolded shape: {x.shape}")
        x = x.contiguous().view(x.size(0), x.size(1), x.size(2) * x.size(3), 8, 8)
        print(f"contiguous shape: {x.shape}")
        x = dct.dct_2d(x, norm=None)
        return x

    def get_temporal_dim(self, window_size):
        return self.config.data.window_size

    def get_encoding_dim(self):
        return self.config.data.shape[1] * self.config.data.shape[2]
