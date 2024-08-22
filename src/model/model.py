import torch
import torch.nn as nn

from src.model.encoder.encoder import Encoder
from src.model.decoder.decoder import Decoder


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config

        self.encoder = Encoder(self.config)
        self.config.model.decoder.temporal_dim = self.encoder.get_temporal_dim()
        self.config.model.decoder.encoding_dim = self.encoder.get_encoding_dim()
        self.decoder = Decoder(self.config)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
