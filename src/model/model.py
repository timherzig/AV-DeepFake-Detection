import torch
import torch.nn as nn

from src.model.encoder.encoder import Encoder
from src.model.decoder.decoder import Decoder


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
