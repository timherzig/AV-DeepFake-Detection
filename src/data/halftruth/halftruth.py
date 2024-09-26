import os
import json
import math
import torch
import random

import numpy as np
import webdataset as wds

from math import floor
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

    if audio.shape[1] != audio_frames:
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
        start3 = floor(audio_len / 2)  # take the middle segment as well
        audio1 = audio[:, start1 : start1 + audio_frames]
        audio2 = audio[:, start2 : start2 + audio_frames]
        audio3 = audio[:, start3 : start3 + audio_frames]

        if audio1.shape[1] != audio_frames:
            audio1 = pad(audio1, (0, audio_frames - audio1.shape[1]), "constant", 0)
        if audio2.shape[1] != audio_frames:
            audio2 = pad(audio2, (0, audio_frames - audio2.shape[1]), "constant", 0)
        if audio3.shape[1] != audio_frames:
            audio3 = pad(audio3, (0, audio_frames - audio3.shape[1]), "constant", 0)

        audio = torch.stack([audio1, audio2, audio3])
        label = torch.tensor([[0.0, 1.0], [0.0, 1.0]])

    else:
        cut_audio = []
        label = []

        for transition in fake_segments:
            for t in transition:
                t = t * sr + audio_frames
                start = floor(t - audio_frames // 2)
                audio_slice = audio[:, start : start + audio_frames]
                if audio_slice.shape[1] != audio_frames:
                    audio_slice = pad(
                        audio_slice,
                        (0, audio_frames - audio_slice.shape[1]),
                        "constant",
                        0,
                    )
                cut_audio.append(audio_slice)
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

        if audio_slice.shape[1] != window_size:
            audio_slice = pad(
                audio_slice, (0, window_size - audio_slice.shape[1]), "constant", 0
            )

        sliced_audio.append(audio_slice)

    audio = torch.stack(sliced_audio)

    label = torch.zeros(audio.shape[0], 2)
    fake_segments = [i for ii in fake_segments for i in ii]

    for t in fake_segments:
        index = floor(t * sr) // step_size + ((window_size // 2) // step_size)
        label[index] = torch.tensor([1.0, 0.0])

    return audio, label


def audio_collate_fn(audio, label, config, sliding_window=False, test=False):
    if not sliding_window:
        if not test:
            audio, label = zip(*[cut_audio(a, l, config) for a, l in zip(audio, label)])
            # normalize audio
            audio = [normalize(i, dim=1) for i in audio]
            audio = torch.stack(audio).squeeze()
            label = torch.tensor(label)
        else:
            audio, label = zip(
                *[cut_audio_test(a, l, config) for a, l in zip(audio, label)]
            )
            audio = [normalize(i, dim=1) for i in audio]
            audio = torch.cat(audio)
            label = torch.cat(label)
    else:
        audio, label = zip(
            *[cut_sliding_window_audio(a, l, config) for a, l in zip(audio, label)]
        )
        # normalize audio
        audio = [normalize(i, dim=1) for i in audio]
        audio = torch.cat(audio).squeeze()
        label = torch.cat(label)

    return audio, label


def halfthruth_collate_fn(batch, config, sliding_window=False, test=False):
    audio, a_info = batch
    audio, _ = zip(*audio)

    if config.model.task == "audio":
        return audio_collate_fn(
            audio,
            [i["audio_fake_segments"] for i in a_info],
            config,
            sliding_window=sliding_window,
            test=test,
        )
    elif config.model.task == "video":
        raise NotImplementedError("HalfTruth is audio only")


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
        raise NotImplementedError(
            "Pre-encoded data is not yet supported for HalfTruth dataset"
        )
    else:
        dataset = (
            wds.WebDataset(tar_paths)
            .decode(
                wds.torch_audio,
                wds.handle_extension("json", json_decoder),
            )
            .to_tuple("wav", "json")
            .batched(config.train.batch_size)
        )

    if train:
        dataset = dataset.shuffle(config.train.shuffle)

    return dataset


def halftruth_get_splits(
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
            os.path.join(root, "dev", i) for i in os.listdir(os.path.join(root, "dev"))
        ]
    else:
        val_parts = [
            os.path.join(root, "dev", f"dev_{str(p).zfill(3)}.tar.gz")
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
