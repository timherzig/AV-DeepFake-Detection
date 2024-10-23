import os
import sys
import math
import json
import torch
import random

import numpy as np
import webdataset as wds

from math import floor
from io import BytesIO
from torch.nn.functional import pad, normalize


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

        label = [0.0, 1.0]
    else:
        transition = (
            random.choice(random.choice(fake_segments)) * sr + audio_frames
        )  # randomly chosen transition corrected for sample rate and the previous padding

        if config.data.center_transition:
            start = floor(transition - audio_frames // 2)
        else:
            start = random.randint(transition - audio_frames + 1, transition - 1)

        audio = audio[:, start : start + audio_frames]
        label = [1.0, 0.0]

    if audio.shape[1] < audio_frames:
        audio = pad(audio, (0, audio_frames - audio.shape[1]), "constant", 0)

    return audio, label


def cut_audio_test(audio, fake_segments, config):
    n_frames = config.data.window_size
    sr = config.data.sr

    audio_len = audio.shape[1]
    audio_frames = n_frames * (sr // config.data.fps)
    audio = pad(audio, (audio_frames, audio_frames), "constant", 0)

    if len(fake_segments) == 0:
        start1 = audio_frames  # In the test case always take the first and last audio segment for consistency
        start2 = audio_len
        start3 = floor(audio_len / 2)  # Take the middle segment
        audio1 = audio[:, start1 : start1 + audio_frames]
        audio2 = audio[:, start2 : start2 + audio_frames]
        audio3 = audio[:, start3 : start3 + audio_frames]

        if audio1.shape[1] < audio_frames:
            audio1 = pad(audio1, (0, audio_frames - audio1.shape[1]), "constant", 0)
        if audio2.shape[1] < audio_frames:
            audio2 = pad(audio2, (0, audio_frames - audio2.shape[1]), "constant", 0)
        if audio3.shape[1] < audio_frames:
            audio3 = pad(audio3, (0, audio_frames - audio3.shape[1]), "constant", 0)

        audio = torch.stack([audio1, audio2, audio3])
        label = torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])

    else:
        cut_audio = []
        label = []
        for transition in fake_segments:
            for t in transition:
                t = t * sr + audio_frames
                start = floor(t - audio_frames // 2)
                audio1 = audio[:, start : start + audio_frames]
                if audio1.shape[1] < audio_frames:
                    audio1 = pad(
                        audio1, (0, audio_frames - audio1.shape[1]), "constant", 0
                    )
                cut_audio.append(audio1)
                label.append([1.0, 0.0])

        audio = torch.stack(cut_audio)
        label = torch.tensor(label)

    audio = audio.squeeze()

    return audio, label


def cut_sliding_window_audio(audio, fake_segments, config, step_size=4):
    window_size = config.data.window_size
    sr = config.data.sr
    fps = config.data.fps

    audio_len = audio.shape[1]
    window_size = window_size * (sr // fps)
    step_size = step_size * (sr // fps)
    audio = pad(audio, (window_size, window_size), "constant", 0)

    sliced_audio = []

    for i in range(0, audio_len + window_size, step_size):
        audio_slice = audio[:, i : i + window_size]

        if audio_slice.shape[1] < window_size:
            audio_slice = pad(
                audio_slice, (0, window_size - audio_slice.shape[1]), "constant", 0
            )

        # normalize slice
        audio_slice = normalize(audio_slice, dim=1)

        sliced_audio.append(audio_slice)

    audio = torch.stack(sliced_audio)

    label = torch.zeros(audio.shape[0], 2)
    label[:, 1] = 1.0
    fake_segments = [i for ii in fake_segments for i in ii]
    fake_segments.append(0.0)

    for t in fake_segments:
        t = t * sr + window_size
        start = floor(t - window_size // 2)
        index = start // step_size

        if index < len(label):
            label[index] = torch.tensor([1.0, 0.0])

    return audio, label


def cut_video(video, fake_segments, config):
    n_frames = config.data.window_size
    fps = config.data.fps

    video_len = video.shape[0]
    video_pad = n_frames

    video = pad(video, (0, 0, 0, 0, 0, 0, video_pad, video_pad), "constant", 0)

    if len(fake_segments) == 0:
        start = random.randint(video_pad, video_len - n_frames - video_pad)
        video = video[start : start + n_frames, :, :, :]

        label = [0.0, 1.0]
    else:
        transition = random.choice(random.choice(fake_segments)) * fps + video_pad

        if config.data.center_transition:
            start = floor(transition - n_frames // 2)
        else:
            start = random.randint(transition - n_frames + 1, transition - 1)

        video = video[start : start + n_frames, :, :, :]
        label = [1.0, 0.0]

    if video.shape[0] < n_frames:
        video = pad(
            video,
            (
                0,
                config.data.shape[0] - video.shape[3],
                0,
                config.data.shape[1] - video.shape[2],
                0,
                config.data.shape[2] - video.shape[1],
                0,
                n_frames - video.shape[0],
            ),
            "constant",
            0,
        )

    return video, label


def cut_video_test(video, fake_segments, config):
    n_frames = config.data.window_size
    fps = config.data.fps
    video_len = video.shape[0]
    video_pad = n_frames

    video = pad(video, (0, 0, 0, 0, 0, 0, video_pad, video_pad), "constant", 0)

    if len(fake_segments) == 0:
        start1 = video_pad
        start2 = video_len
        video1 = video[start1 : start1 + n_frames, :, :, :]
        video2 = video[start2 : start2 + n_frames, :, :, :]

        if video1.shape[0] < n_frames:
            video1 = pad(
                video1,
                (
                    0,
                    config.data.shape[0] - video1.shape[3],
                    0,
                    config.data.shape[1] - video1.shape[2],
                    0,
                    config.data.shape[2] - video1.shape[1],
                    0,
                    n_frames - video1.shape[0],
                ),
                "constant",
                0,
            )
        if video2.shape[0] < n_frames:
            video2 = pad(
                video2,
                (
                    0,
                    config.data.shape[0] - video2.shape[3],
                    0,
                    config.data.shape[1] - video2.shape[2],
                    0,
                    config.data.shape[2] - video2.shape[1],
                    0,
                    n_frames - video2.shape[0],
                ),
                "constant",
                0,
            )

        video = torch.stack([video1, video2])
        label = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    else:
        cut_video = []
        label = []
        for transition in fake_segments:
            for t in transition:
                t = t * fps + video_pad
                start = floor(t - n_frames // 2)
                video1 = video[start : start + n_frames, :, :, :]
                if video1.shape[0] < n_frames:
                    video1 = pad(
                        video1,
                        (
                            0,
                            config.data.shape[0] - video1.shape[3],
                            0,
                            config.data.shape[1] - video1.shape[2],
                            0,
                            config.data.shape[2] - video1.shape[1],
                            0,
                            n_frames - video1.shape[0],
                        ),
                        "constant",
                        0,
                    )
                cut_video.append(video1)
                label.append([1.0, 0.0])

        video = torch.stack(cut_video)
        label = torch.tensor(label)

    return video, label


def cut_sliding_window_video(video, fake_segments, config, step_size=4):
    window_size = config.data.window_size
    fps = config.data.fps

    video_len = video.shape[0]
    video = pad(video, (0, 0, 0, 0, 0, 0, window_size, window_size), "constant", 0)

    sliced_video = []

    for i in range(0, video_len + window_size, step_size):
        video_slice = video[i : i + window_size, :, :, :]
        if video_slice.shape[0] < window_size:
            video_slice = pad(
                video_slice,
                (
                    0,
                    config.data.shape[0] - video_slice.shape[3],
                    0,
                    config.data.shape[1] - video_slice.shape[2],
                    0,
                    config.data.shape[2] - video_slice.shape[1],
                    0,
                    window_size - video_slice.shape[0],
                ),
                "constant",
                0,
            )
        sliced_video.append(video_slice)

    video = torch.stack(sliced_video)

    label = torch.zeros(video.shape[0], 2)
    label[:, 1] = 1.0
    fake_segments = [i for ii in fake_segments for i in ii]
    fake_segments.append(0.0)

    for t in fake_segments:
        t = t * fps + window_size
        start = floor(t - window_size // 2)
        index = start // step_size

        if index < len(label):
            label[index] = torch.tensor([1.0, 0.0])

    return video, label


def cut_audio_video(audio, video, audio_fake_segments, video_fake_segments, config):
    n_frames = config.data.window_size
    sr = config.data.sr
    fps = config.data.fps

    audio_len = audio.shape[1]
    audio_frames = n_frames * (sr // fps)
    audio = pad(audio, (audio_frames, audio_frames), "constant", 0)

    video_len = video.shape[0]
    video_pad = n_frames
    video = pad(video, (0, 0, 0, 0, 0, 0, video_pad, video_pad), "constant", 0)

    if len(audio_fake_segments) == 0 and len(video_fake_segments) == 0:
        start_time = random.randint(video_pad, video_len - n_frames - video_pad)
        video = video[start_time : start_time + n_frames, :, :, :]
        start_time = start_time * (sr // fps)
        audio = audio[:, start_time : start_time + audio_frames]
        label = [[0.0, 1.0], [0.0, 1.0]]  # [a, v]

    elif len(audio_fake_segments) == 0:
        start_time = random.choice(random.choice(video_fake_segments)) * fps + video_pad
        if config.data.center_transition:
            start_time = floor(start_time - n_frames // 2)
        else:
            start_time = random.randint(start_time - n_frames + 1, start_time - 1)
        video = video[start_time : start_time + n_frames, :, :, :]
        start_time = start_time * (sr // fps)
        audio = audio[:, start_time : start_time + audio_frames]
        label = [[0.0, 1.0], [1.0, 0.0]]

    elif len(video_fake_segments) == 0:
        transition = (
            random.choice(random.choice(audio_fake_segments)) * sr + audio_frames
        )
        if config.data.center_transition:
            start_time = floor(transition - n_frames // 2)
        else:
            start_time = random.randint(transition - n_frames + 1, transition - 1)
        audio = audio[:, start_time : start_time + audio_frames]
        start_time = start_time // (sr // fps)
        video = video[start_time : start_time + n_frames, :, :, :]
        label = [[1.0, 0.0], [0.0, 1.0]]

    else:
        transition = (
            random.choice(random.choice(audio_fake_segments)) * sr + audio_frames
        )
        if config.data.center_transition:
            start_time = floor(transition - n_frames // 2)
        else:
            start_time = random.randint(transition - n_frames + 1, transition - 1)
        audio = audio[:, start_time : start_time + audio_frames]
        start_time = start_time // (sr // fps)
        video = video[start_time : start_time + n_frames, :, :, :]
        label = [[1.0, 0.0], [1.0, 0.0]]

    if audio.shape[1] < audio_frames:
        audio = pad(audio, (0, audio_frames - audio.shape[1]), "constant", 0)

    if video.shape[0] < n_frames:
        video = pad(
            video,
            (
                0,
                config.data.shape[0] - video.shape[3],
                0,
                config.data.shape[1] - video.shape[2],
                0,
                config.data.shape[2] - video.shape[1],
                0,
                n_frames - video.shape[0],
            ),
            "constant",
            0,
        )

    return audio, video, label


def cut_audio_video_test(
    audio, video, audio_fake_segments, video_fake_segments, config
):
    pass


def cut_sliding_window_audio_video(
    audio, video, audio_fake_segments, video_fake_segments, config
):
    pass


def audio_collate_fn(audio, label, config, sliding_window=False, test=False):
    if not sliding_window:
        if not test:
            audio, label = zip(*[cut_audio(a, l, config) for a, l in zip(audio, label)])
            # normalize audio
            audio = [normalize(a, dim=1) for a in audio]
            audio = torch.stack(audio).squeeze()
            label = torch.tensor(label)
        else:
            audio, label = zip(
                *[cut_audio_test(a, l, config) for a, l in zip(audio, label)]
            )
            # normalize audio
            audio = [normalize(a, dim=1) for a in audio]
            audio = torch.cat(audio)
            label = torch.cat(label)
    else:
        audio, label = zip(
            *[
                cut_sliding_window_audio(a, l, config, step_size=config.data.step_size)
                for a, l in zip(audio, label)
            ]
        )
        audio = torch.cat(audio).squeeze()
        label = torch.cat(label)

    return audio, label


def video_collate_fn(video, label, config, sliding_window=False, test=False):
    if not sliding_window:
        if not test:
            video, label = zip(*[cut_video(v, l, config) for v, l in zip(video, label)])
            video = torch.stack(video).squeeze()
            label = torch.tensor(label)
        else:
            video, label = zip(
                *[cut_video_test(v, l, config) for v, l in zip(video, label)]
            )
            video = torch.cat(video)
            label = torch.cat(label)
    else:
        video, label = zip(
            *[
                cut_sliding_window_video(v, l, config, step_size=config.data.step_size)
                for v, l in zip(video, label)
            ]
        )
        video = torch.cat(video).squeeze()
        label = torch.cat(label)

    video = video.type(torch.float32).permute(0, 4, 1, 2, 3)

    return video, label


def audio_video_collate_fn(
    audio, video, av_info, config, sliding_window=False, test=False
):
    audio_fake_segments = [i["audio_fake_segments"] for i in av_info]
    video_fake_segments = [i["video_fake_segments"] for i in av_info]

    if not sliding_window:
        if not test:
            audio, video, label = zip(
                *[
                    cut_audio_video(a, v, afs, vfs, config)
                    for a, v, afs, vfs in zip(
                        audio,
                        video,
                        audio_fake_segments,
                        video_fake_segments,
                    )
                ]
            )
            audio = [normalize(a, dim=1) for a in audio]
            audio = torch.stack(audio).squeeze()
            video = torch.stack(video).squeeze()
            label = torch.tensor(label)
        else:
            audio, video, label = zip(
                *[
                    cut_audio_video_test(a, v, afs, vfs, config)
                    for a, v, afs, vfs in zip(
                        audio,
                        video,
                        audio_fake_segments,
                        video_fake_segments,
                    )
                ]
            )
            audio = [normalize(a, dim=1) for a in audio]
            audio = torch.cat(audio)
            video = torch.cat(video)
            label = torch.cat(label)
    else:
        audio, video, label = zip(
            *[
                cut_sliding_window_audio_video(a, v, afs, vfs, config)
                for a, v, afs, vfs in zip(
                    audio,
                    video,
                    audio_fake_segments,
                    video_fake_segments,
                )
            ]
        )
        audio = torch.cat(audio).squeeze()
        video = torch.cat(video).squeeze()
        label = torch.cat(label)

    video = video.type(torch.float32).permute(0, 4, 1, 2, 3)

    return (audio, video), label


def av1m_collate_fn(batch, config, sliding_window=False, test=False):
    x, av_info = batch
    video, audio, _ = zip(*x)

    if config.model.task == "audio":
        return audio_collate_fn(
            audio,
            [i["audio_fake_segments"] for i in av_info],
            config,
            sliding_window=sliding_window,
            test=test,
        )
    elif config.model.task == "video":
        return video_collate_fn(
            video,
            [i["video_fake_segments"] for i in av_info],
            config,
            sliding_window=sliding_window,
            test=test,
        )
    elif config.model.task == "audio-video":
        return audio_video_collate_fn(
            audio,
            video,
            av_info,
            config,
            sliding_window=sliding_window,
            test=test,
        )


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
