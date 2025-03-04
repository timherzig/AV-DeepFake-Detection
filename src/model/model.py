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

        if self.config.model.encoder_freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x, return_encoding=False):
        x = self.encoder(x)
        x = self.decoder(x, return_encoding=return_encoding)

        return x

    def stage1(self, x):
        x = self.encoder(x)
        return x

    def stage2(self, x, return_encoding=False):
        x = self.decoder(x, return_encoding=return_encoding)
        return x

    def get_encoding_dim(self):
        return self.config.model.decoder.encoding_dim

    def get_temporal_dim(self):
        return self.config.model.decoder.temporal_dim
