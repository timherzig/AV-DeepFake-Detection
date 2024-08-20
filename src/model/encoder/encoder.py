import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config

        self.encoder = None

        # AUDIO
        if config.encoder == "WavLM":
            from src.model.encoder.wavlm.WavLM import WavLM, WavLMConfig

            ckpt = torch.load(config.encoder.pretrained_path, map_location="cpu")
            cfg = WavLMConfig(ckpt["cfg"])
            self.encoder = WavLM(cfg)

        # VIDEO
        elif config.encoder == "VideoMamba":
            from src.model.encoders.videomamba.videomamba import (
                videomamba_middle,
                videomamba_small,
                videomamba_tiny,
            )

            if self.encoder_config.version == "middle":
                self.encoder = videomamba_middle(self.config.encoder_config)
            elif self.video_encoder_config.version == "small":
                self.encoder = videomamba_small(self.config.encoder_config)
            elif self.video_encoder_config.version == "tiny":
                self.encoder = videomamba_tiny(self.config.encoder_config)
            else:
                raise ValueError("Invalid version for the videomamba encoder")

    def forward(self, x):
        return self.encoder(x)
