import torch
import torch.nn as nn

from torch.nn.functional import pad
from collections import OrderedDict


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config

        self.encoder = None
        self.succeeding_layers = None

        # AUDIO
        if self.config.model.task == "audio":
            if self.config.model.encoder.name.lower() == "wavlm":
                from src.model.encoder.wavlm.WavLM import WavLM, WavLMConfig

                ckpt = torch.load(
                    config.model.encoder.pretrained_path, map_location="cpu"
                )
                cfg = WavLMConfig(ckpt["cfg"])
                self.encoder = WavLM(cfg)

        # VIDEO
        if self.config.model.task == "video":
            if self.config.model.encoder.name.lower() == "videomamba":
                from src.model.encoder.videomamba.videomamba import (
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

            elif self.config.model.encoder.name.lower() == "videoswin":
                from src.model.encoder.videoswin.video_swin_transformer import (
                    SwinTransformer3D,
                )

                self.encoder = SwinTransformer3D(
                    embed_dim=128,
                    depths=[2, 2, 18, 2],
                    num_heads=[4, 8, 16, 32],
                    patch_size=(2, 4, 4),
                    window_size=(16, 7, 7),
                    drop_path_rate=0.4,
                    patch_norm=True,
                    config=self.config,
                )

                # Currently only works with: "encoder_checkpoints/videoswin/swin_base_patch244_window1677_sthv2.pth"
                checkpoint = torch.load(self.config.model.encoder.pretrained_path)

                new_state_dict = OrderedDict()
                for k, v in checkpoint["state_dict"].items():
                    if "backbone" in k:
                        name = k[9:]
                        new_state_dict[name] = v

                self.encoder.load_state_dict(new_state_dict)

            elif self.config.model.encoder.name.lower() == "dct":
                from src.model.encoder.dct.dct import DCT

                self.encoder = DCT(self.config)

            elif self.config.model.encoder.name.lower() == "jpeg":
                from src.model.encoder.jpeg.jpeg import JPEG

                self.encoder = JPEG(self.config)

        if self.config.model.task == "audio-video":
            # AUDIO
            audio_encoder = None

            if self.config.model.encoder.audio.name.lower() == "wavlm":
                from src.model.encoder.wavlm.WavLM import WavLM, WavLMConfig

                ckpt = torch.load(
                    config.model.encoder.audio.pretrained_path, map_location="cpu"
                )
                cfg = WavLMConfig(ckpt["cfg"])
                audio_encoder = WavLM(cfg)
            else:
                raise NotImplementedError

            # VIDEO
            video_encoder = None

            if self.config.model.encoder.video.name.lower() == "videomamba":
                from src.model.encoder.videomamba.videomamba import (
                    videomamba_middle,
                    videomamba_small,
                    videomamba_tiny,
                )

                if self.config.model.encoder.video.version == "middle":
                    video_encoder = videomamba_middle(self.config.model.encoder.video)
                elif self.config.model.encoder.video.version == "small":
                    video_encoder = videomamba_small(self.config.model.encoder.video)
                elif self.config.model.encoder.video.version == "tiny":
                    video_encoder = videomamba_tiny(self.config.model.encoder.video)
                else:
                    raise ValueError("Invalid version for the videomamba encoder")
            else:
                raise NotImplementedError

            self.encoder = nn.ModuleList([audio_encoder, video_encoder])

    def forward(self, x):
        if self.config.model.task == "audio-video":
            audio_x, video_x = x

            audio_x = self.encoder[0](audio_x)
            video_x = self.encoder[1](video_x)

            if audio_x.shape[1] > video_x.shape[1]:
                video_x = pad(video_x, (0, 0, 0, audio_x.shape[1] - video_x.shape[1]))
            elif audio_x.shape[1] < video_x.shape[1]:
                audio_x = pad(audio_x, (0, 0, 0, video_x.shape[1] - audio_x.shape[1]))

            audio_x = audio_x.permute(2, 0, 1)
            video_x = video_x.permute(2, 0, 1)

            x = torch.cat([audio_x, video_x], dim=0)
            x = x.permute(1, 2, 0)
            return x

        if self.config.model.encoder.name.lower() == "wavlm":
            return self.encoder(x, output_layer=self.config.model.encoder.output_layer)
        elif self.config.model.encoder.name.lower() == "videomamba":
            return self.encoder(x)
        elif self.config.model.encoder.name.lower() == "videoswin":
            return self.encoder(x)
        elif self.config.model.encoder.name.lower() == "dct":
            return self.encoder(x)
        elif self.config.model.encoder.name.lower() == "jpeg":
            return self.encoder(x)

        return self.encoder(x)

    def get_encoding_dim(self):
        if self.config.model.task == "audio-video":
            return (
                self.encoder[0].get_encoding_dim() + self.encoder[1].get_encoding_dim()
            )

        return self.encoder.get_encoding_dim()

    def get_temporal_dim(self):
        if self.config.model.task == "audio-video":
            return max(
                self.encoder[0].get_temporal_dim(
                    window_size=self.config.data.window_size
                ),
                self.encoder[1].get_temporal_dim(
                    window_size=self.config.data.window_size
                ),
            )

        return self.encoder.get_temporal_dim(window_size=self.config.data.window_size)
