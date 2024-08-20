import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config

        self.decoder = None

    def forward(self, x):
        return self.decoder(x)
