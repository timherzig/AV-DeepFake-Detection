import torch

from torchvision.transforms import Grayscale

import torch.nn as nn
import torch_dct as dct


def zigzag(matrix):
    """Zigzag function.
    Args:
        matrix: Input tensor. (B, T, 784, H, W)
    """

    rows, cols = matrix.size(3), matrix.size(4)

    result = []

    for diag in range(rows + cols - 1):
        if diag % 2 == 0:
            # Even diagonals traverse from bottom-left to top-right
            row = min(diag, rows - 1)
            col = diag - row
            while row >= 0 and col < cols:
                result.append(matrix[:, :, :, row, col])
                row -= 1
                col += 1
        else:
            # Odd diagonals traverse from top-right to bottom-left
            col = min(diag, cols - 1)
            row = diag - col
            while col >= 0 and row < rows:
                result.append(matrix[:, :, :, row, col])
                row += 1
                col -= 1

    result = torch.stack(result, dim=3)

    return result


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
        x = x.contiguous().view(x.size(0), x.size(1), x.size(2) * x.size(3), 8, 8)
        x = dct.dct_2d(x, norm=None)
        x = zigzag(x)
        # Remove DC coefficients and truncate the last AC coefficients
        x = x[:, :, :, 1 : -self.config.model.encoder.remove_last_ac]
        x = x.reshape(x.size(0), x.size(1), x.size(2) * x.size(3))
        return x

    def get_temporal_dim(self, window_size):
        return self.config.data.window_size

    def get_encoding_dim(self):
        return (8 * 8 - 1 - self.config.model.encoder.remove_last_ac) * 784
