import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config

        self.decoder = None

        if config.model.decoder.name.lower() == "mlp":
            from src.model.decoder.mlp.mlp import MLP

            self.decoder = MLP(config)
        elif config.model.decoder.name.lower() == "gmlp":
            raise NotImplementedError
        elif config.model.decoder.name.lower() == "aasist":
            from src.model.decoder.aasist.aasist import AASIST

            self.decoder = AASIST(config)

    def forward(self, x, return_encoding=False):
        return self.decoder(x, return_encoding=return_encoding)
