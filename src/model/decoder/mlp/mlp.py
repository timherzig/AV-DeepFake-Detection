import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config

        self.linear1 = nn.Linear(
            config.model.decoder.encoding_dim, config.model.decoder.hidden_size
        )
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(
            config.model.decoder.hidden_size, config.model.decoder.output_size
        )

    def forward(self, x):
        print(f"decoder input shape: {x.shape}")
        x = self.linear1(x)
        print(f"linear1 output shape: {x.shape}")
        x = self.relu(x)
        print(f"relu output shape: {x.shape}")
        x = self.linear2(x)
        print(f"linear2 output shape: {x.shape}")
        return x
