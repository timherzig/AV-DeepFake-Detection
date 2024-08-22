import os
import math
import json
import torch
import random

import numpy as np
import webdataset as wds

from math import floor
from io import BytesIO
from librosa.feature import melspectrogram
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad


# def cut_segments(video, audio, video_fake_segments, audio_fake_segments, config):
#     n_frames = config.data.window_size
#     fps = config.data.fps
#     sr = config.data.sr

#     video_len = video.shape[0]
#     audio_len = audio.shape[0]

#     video_pad = n_frames
#     audio_pad = n_frames * (sr // fps)

#     video = pad(video, (0, 0, 0, 0, 0, 0, video_pad, video_pad), "constant", 0)
#     audio = pad(audio, (0, 0, audio_pad, audio_pad), "constant", 0)

#     video_fake_segments = [x for xs in video_fake_segments for x in xs]
#     audio_fake_segments = [x for xs in audio_fake_segments for x in xs]

#     fake_video = 0
#     fake_audio = 0

#     if len(video_fake_segments) == 0 and len(audio_fake_segments) == 0:
#         start = (
#             random.randint(video_pad, video.shape[0] - n_frames - video_pad)
#             if video_len > n_frames
#             else video_pad
#         )
#         video = video[start : start + n_frames, :, :, :]
#         audio = audio[start * (sr // fps) : (start + n_frames) * (sr // fps), :]
#     else:
#         if len(video_fake_segments) > 0 and len(audio_fake_segments) > 0:
#             video_fake = bool(random.getrandbits(1))
#             fake_video = 1
#             fake_audio = 1
#             if video_fake:
#                 transition = (
#                     math.floor(random.choice(video_fake_segments) * fps) + video_pad
#                 )
#             else:
#                 transition = (
#                     math.floor(random.choice(audio_fake_segments) * fps) + video_pad
#                 )
#         elif len(video_fake_segments) > 0 and len(audio_fake_segments) == 0:
#             transition = (
#                 math.floor(random.choice(video_fake_segments) * fps) + video_pad
#             )
#             fake_video = 1
#         elif len(video_fake_segments) == 0 and len(audio_fake_segments) > 0:
#             transition = (
#                 math.floor(random.choice(audio_fake_segments) * fps) + video_pad
#             )
#             fake_audio = 1

#         # transition is the frame in which the transition happens

#         if config.data.center_transition:
#             start = transition - n_frames // 2
#         else:
#             start = random.randint(
#                 transition - n_frames + 1,
#                 min(transition - 1, video.shape[0] - n_frames),
#             )

#         video = video[start : start + n_frames, :, :, :]
#         audio = audio[start * (sr // fps) : (start + n_frames) * (sr // fps), :]

#         if video.shape[0] < n_frames:
#             video = pad(
#                 video,
#                 (0, 0, 0, 0, 0, 0, 0, n_frames - video.shape[0]),
#                 "constant",
#                 0,
#             )
#         if audio.shape[0] < n_frames * (sr // fps):
#             audio = pad(
#                 audio,
#                 (0, 0, 0, n_frames * (sr // fps) - audio.shape[0]),
#                 "constant",
#                 0,
#             )

#     # if config.model.encoder.name == "spectogram":
#     #     audio = melspectrogram(
#     #         y=audio.T.numpy(),
#     #         sr=sr,
#     #         # n_mels=config.model.audio_encoder.n_mels,
#     #         hop_length=config.model.audio_encoder.hop_length,
#     #         n_fft=config.model.audio_encoder.n_fft,
#     #     )
#     #     audio = torch.tensor(audio).float()

#     fake_video = [1.0, 0.0] if fake_video == 1 else [0.0, 1.0]
#     fake_audio = [1.0, 0.0] if fake_audio == 1 else [0.0, 1.0]

#     return video, fake_video, audio, fake_audio


def cut_audio(audio, fake_segments, config):
    n_frames = config.data.window_size
    sr = config.data.sr

    audio_len = audio.shape[1]
    audio_frames = n_frames * (sr // config.data.fps)
    audio = pad(
        audio, (audio_frames, audio_frames), "constant", 0
    )  # pad audio in case the chosen fake segment is at the beginning or end of the audio

    if len(fake_segments) == 0:
        start = random.randint(audio_frames, audio_len + audio_frames)
        audio = audio[:, start : start + audio_frames]

        return audio, [0.0, 1.0]
    else:
        transition = (
            random.choice(random.choice(fake_segments)) * sr + audio_frames
        )  # randomly chosen transition corrected for sample rate and the previous padding

        if config.data.center_transition:
            start = floor(transition - audio_frames // 2)
        else:
            start = random.randint(transition - audio_frames + 1, transition - 1)

        audio = audio[:, start : start + audio_frames]

        return audio, [1.0, 0.0]


def cut_video(video, fake_segments, config):
    n_frames = config.data.window_size
    fps = config.data.fps

    video_len = video.shape[0]
    video_pad = n_frames

    video = pad(video, (0, 0, 0, 0, 0, 0, video_pad, video_pad), "constant", 0)

    if len(fake_segments) == 0:
        start = random.randint(video_pad, video_len - n_frames - video_pad)
        video = video[start : start + n_frames, :, :, :]

        return video, [0.0, 1.0]
    else:
        transition = random.choice(random.choice(fake_segments)) * fps + video_pad

        if config.data.center_transition:
            start = floor(transition - n_frames // 2)
        else:
            start = random.randint(transition - n_frames + 1, transition - 1)

        video = video[start : start + n_frames, :, :, :]

        return video, [1.0, 0.0]


def audio_collate_fn(audio, label, config):
    audio, label = zip(*[cut_audio(a, l, config) for a, l in zip(audio, label)])
    audio = torch.stack(audio).squeeze()
    label = torch.tensor(label)
    return audio, label


def video_collate_fn(video, label, config):
    video, label = zip(*[cut_video(v, l, config) for v, l in zip(video, label)])
    video = torch.stack(video).squeeze()
    label = torch.tensor(label)
    return video, label


def av1m_collate_fn(batch, config):
    x, av_info = batch
    video, audio, _ = zip(*x)

    if config.model.task == "audio":
        return audio_collate_fn(
            audio, [i["audio_fake_segments"] for i in av_info], config
        )
    elif config.model.task == "video":
        return video_collate_fn(
            video, [i["video_fake_segments"] for i in av_info], config
        )

    # audio = [a.T for a in audio]

    # info = [
    #     {
    #         "id": "/".join(i_info["video_path"].split("/")[4:9]),
    #         "video_fake": True if "fake_video" in i_info["video_path"] else False,
    #         "audio_fake": True if "fake_audio" in i_info["video_path"] else False,
    #         "video_fake_segments": i_info["video_fake_segments"],
    #         "audio_fake_segments": i_info["audio_fake_segments"],
    #         "video_frames": i_info["video_frames"],
    #         "audio_frames": i_info["audio_frames"],
    #     }
    #     for i_info in av_info
    # ]

    # video, video_label, audio, audio_label = zip(
    #     *[
    #         cut_segments(
    #             v, a, i["video_fake_segments"], i["audio_fake_segments"], config
    #         )
    #         for v, a, i in zip(video, audio, info)
    #     ]
    # )

    # if config.model.task == "audio":
    #     x = torch.stack(audio).squeeze()
    #     x_label = torch.tensor(audio_label)
    # elif config.model.task == "video":
    #     x = torch.stack(video).float()
    #     x_label = torch.tensor(video_label)
    # else:
    #     # Not focus for now
    #     x = (torch.stack(video), torch.stack(audio))
    #     x_label = (torch.tensor(video_label), torch.tensor(audio_label))

    # return (x, x_label), info


def npz_decoder(data):
    data = np.load(BytesIO(data))
    video = torch.from_numpy(data["video"])
    audio = torch.from_numpy(data["audio"])
    label = torch.from_numpy(data["label"])
    return video, audio, label


def json_decoder(data):
    data = json.loads(data)
    return data


def get_data(tar_paths, config, train=True):
    # Get the web-dataset for the specified tar-files
    # Parameters
    # ----------
    # tar_paths : list
    #     List of paths to the tar-files
    # config : OmegaConf
    #     Configuration object
    # train : bool
    #     Whether the dataset is for training or not
    # Returns
    # -------
    # dataset : wds.WebDataset
    #     WebDataset object

    if not config.model.online_encoding:
        dataset = (
            wds.WebDataset(tar_paths)
            .decode(
                wds.handle_extension("npz", npz_decoder),
                wds.handle_extension("json", json_decoder),
            )
            .to_tuple("npz", "json")
            .batched(config.train.batch_size)
        )
    else:
        dataset = (
            wds.WebDataset(tar_paths)
            .decode(
                wds.torch_video,
                wds.handle_extension("json", json_decoder),
            )
            .to_tuple("mp4", "json")
            .batched(config.train.batch_size)
        )

    if train:
        dataset = dataset.shuffle(config.train.shuffle)

    return dataset


def av1m_get_splits(
    root: str,
    config,
    train_parts="all",
    val_parts="all",
    test_parts="all",
):
    # Get the datasets for the specified splits
    # Parameters
    # ----------
    # root : str
    #     Root directory of the dataset
    # config : OmegaConf
    #     Configuration object
    # train_parts : str or list, optional
    #     Parts of the dataset to use for training, by default "all"
    # val_parts : str or list, optional
    #     Parts of the dataset to use for validation, by default "all"
    # test_parts : str or list, optional
    #     Parts of the dataset to use for testing, by default "all"
    #
    # Returns
    # -------
    # tuple
    #     Tuple containing train, validation, and test datasets along with their lengths

    # the amount of samples in one tar-file as this is different due to pre-encoded data is larger
    if config.model.online_encoding:
        tar_size = 1000
    else:
        tar_size = 64

    if train_parts == "all":
        train_parts = [
            os.path.join(root, "train", i)
            for i in os.listdir(os.path.join(root, "train"))
        ]
    else:
        train_parts = [
            os.path.join(root, "train", f"train_{str(p).zfill(3)}.tar.gz")
            for p in train_parts
        ]

    if val_parts == "all":
        val_parts = [
            os.path.join(root, "val", i) for i in os.listdir(os.path.join(root, "val"))
        ]
    else:
        val_parts = [
            os.path.join(root, "val", f"val_{str(p).zfill(3)}.tar.gz")
            for p in val_parts
        ]

    if test_parts == "all":
        test_parts = [
            os.path.join(root, "test", i)
            for i in os.listdir(os.path.join(root, "test"))
        ]
    else:
        test_parts = [
            os.path.join(root, "test", f"test_{str(p).zfill(3)}.tar.gz")
            for p in test_parts
        ]

    # limit the amount of samples to the specified size (-1 means all samples)
    if config.data.train_size > 0:
        train_parts = train_parts[: math.ceil(config.data.train_size / tar_size)]
    if config.data.val_size > 0:
        val_parts = val_parts[: math.ceil(config.data.val_size / tar_size)]
    if config.data.test_size > 0:
        test_parts = test_parts[: math.ceil(config.data.test_size / tar_size)]

    train_len = len(train_parts) * tar_size
    val_len = len(val_parts) * tar_size
    test_len = len(test_parts) * tar_size

    train = get_data(train_parts, config) if len(train_parts) > 0 else None
    val = get_data(val_parts, config, train=False) if len(val_parts) > 0 else None
    test = get_data(test_parts, config, train=False) if len(test_parts) > 0 else None

    return (train, train_len), (val, val_len), (test, test_len)
