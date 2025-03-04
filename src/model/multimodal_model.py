import torch
from torch import nn
from src.model.model import Model

from src.model.bidirectional_cross_attention import BidirectionalCrossAttention


class Multimodal_Model(nn.Module):
    def __init__(
        self,
        config,
        audio_config,
        video_config,
        audio_weights=None,
        video_weights=None,
        conv_fusion=False,
    ):
        super(Multimodal_Model, self).__init__()

        self.config = config
        self.audio_config = audio_config
        self.video_config = video_config

        self.conv_fusion = conv_fusion

        self.audio_model = Model(audio_config)
        self.video_model = Model(video_config)

        if audio_weights is not None:
            self.audio_model.load_state_dict(audio_weights)
            print(f"Loaded audio weights")
            for param in self.audio_model.parameters():
                param.requires_grad = False
        if video_weights is not None:
            self.video_model.load_state_dict(video_weights)
            print(f"Loaded video weights")
            for param in self.video_model.parameters():
                param.requires_grad = False

        if self.config.model.crossattention:
            self.cross_attention = BidirectionalCrossAttention(
                dim=160, heads=8, dim_head=64, context_dim=160
            )
            # self.cross_attention_v = BidirectionalCrossAttention(
            #     dim=160, heads=8, dim_head=64, context_dim=160
            # )

        if self.conv_fusion:
            self.conv = nn.Conv1d(2, 1, 1)
            self.lin1 = nn.Linear(160, 64)
            self.lin2 = nn.Linear(64, 4)
        else:
            self.a_fc = nn.Linear(320, 160)
            self.a_lin = nn.Linear(160, 80)
            self.a_out_layer = nn.Linear(80, 2)

            self.v_fc = nn.Linear(320, 160)
            self.v_lin = nn.Linear(160, 80)
            self.v_out_layer = nn.Linear(80, 2)

    def forward(self, x):
        a, v = x

        a = self.audio_model(a, return_encoding=True).unsqueeze(1)
        v = self.video_model(v, return_encoding=True).unsqueeze(1)

        if self.config.model.crossattention:
            ca, cv = self.cross_attention(a, v)

        av = torch.cat((ca, cv), dim=1)

        if self.conv_fusion:
            x = self.conv(av).squeeze(1)
            x = torch.relu(self.lin1(x))
            x = self.lin2(x)
            a, v = torch.split(x, 2, dim=1)

            return (a, v)

        else:
            av = av.flatten(1)

            a = self.a_fc(av)
            a = torch.relu(self.a_lin(a))
            # a = self.a_lin(a)
            a = self.a_out_layer(a)

            v = self.v_fc(av)
            v = torch.relu(self.v_lin(v))
            # v = self.v_lin(v)
            v = self.v_out_layer(v)

            return (a, v)
