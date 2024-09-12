import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config

        self.encoder = None

        # AUDIO
        if self.config.model.encoder.name.lower() == "wavlm":
            from src.model.encoder.wavlm.WavLM import WavLM, WavLMConfig

            ckpt = torch.load(config.model.encoder.pretrained_path, map_location="cpu")
            cfg = WavLMConfig(ckpt["cfg"])
            self.encoder = WavLM(cfg)

        # VIDEO
        elif self.config.model.encoder.name.lower() == "videomamba":
            from src.model.encoders.videomamba.videomamba import (
                videomamba_middle,
                videomamba_small,
                videomamba_tiny,
            )

            if self.config.model.encoder.version == "middle":
                self.encoder = videomamba_middle(self.config.model.encoder)
            elif self.config.model.encoder.version == "small":
                self.encoder = videomamba_small(self.config.model.encoder)
            elif self.config.model.encoder.version == "tiny":
                self.encoder = videomamba_tiny(self.config.model.encoder)
            else:
                raise ValueError("Invalid version for the videomamba encoder")

    def forward(self, x):
        if self.config.model.encoder.name.lower() == "wavlm":
            return self.encoder(x, output_layer=self.config.model.encoder.output_layer)
        elif self.config.model.encoder.name.lower() == "videomamba":
            return self.encoder(x)

        return self.encoder(x)

    def get_encoding_dim(self):
        return self.encoder.get_encoding_dim()

    def get_temporal_dim(self):
        return self.encoder.get_temporal_dim(window_size=self.config.data.window_size)
