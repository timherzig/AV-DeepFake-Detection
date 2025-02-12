import torch
from torch import nn
from src.model.model import Model


class Multimodal_Model(nn.Module):
    def __init__(
        self, audio_config, video_config, audio_weights=None, video_weights=None
    ):
        super(Multimodal_Model, self).__init__()

        self.audio_config = audio_config
        self.video_config = video_config

        self.audio_model = Model(audio_config)
        self.video_model = Model(video_config)

        if audio_weights is not None:
            self.audio_model.load_state_dict(audio_weights)
        if video_weights is not None:
            self.video_model.load_state_dict(video_weights)

        self.conv = nn.Conv1d(2, 1, 1)
        self.lin1 = nn.Linear(160, 64)
        self.lin2 = nn.Linear(64, 4)

    def forward(self, x):
        a, v = x

        with torch.no_grad():
            a = self.audio_model(a, return_encoding=True).unsqueeze(1)
            v = self.video_model(v, return_encoding=True).unsqueeze(1)

        av = torch.cat((a, v), dim=1)

        x = self.conv(av).squeeze(1)
        x = torch.relu(self.lin1(x))
        x = self.lin2(x)
        x = x.reshape(-1, 2, 2)

        return x
