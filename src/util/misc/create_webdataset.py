import os
import json
import shutil
import tarfile
import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.io import wavfile
from sklearn.utils import shuffle


def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


def process_split_partialspoof(
    root, new_root, split, temporal_resolution=0.04, tar_size=1000
):
    audio_root = os.path.join(root, split, "con_wav")
    segment_labels = os.path.join(
        root, "segment_labels", f"{split}_seglab_{str(temporal_resolution)}.npy"
    )

    new_root = os.path.join(new_root, split)
    if os.path.exists(new_root):
        shutil.rmtree(new_root)
        os.makedirs(new_root)

    segment_labels = np.load(segment_labels, allow_pickle=True).item()

    part = 0
    for i, audio_file in enumerate(tqdm(os.listdir(audio_root))):
        out_dir = os.path.join(new_root, f"{split}_{str(part).zfill(3)}")
        os.makedirs(out_dir, exist_ok=True)

        audio_id = audio_file.split(".")[0]
        label = segment_labels[audio_id]

        audio_frames = wavfile.read(os.path.join(audio_root, audio_file))[1].size

        shutil.copy(os.path.join(audio_root, audio_file), out_dir)

        new_label = []
        cur_seg = []

        last_label = label[0]
        if last_label == 1:
            cur_seg.append(0.00)
        for i in range(1, len(label)):
            if label[i] != last_label:
                time_stamp = i * temporal_resolution
                cur_seg.append(time_stamp)

                if int(last_label) == 1 and int(label[i]) == 0 and len(cur_seg) == 2:
                    new_label.append(cur_seg)
                    cur_seg = []

            last_label = label[i]

        label_df = {
            "audio_path": os.path.join(out_dir, audio_file),
            "audio_fake_segments": new_label,
            "audio_frames": audio_frames,
        }

        with open(f"{os.path.join(out_dir, audio_id)}.json", "w") as outfile:
            json.dump(label_df, outfile)

        if (
            len(os.listdir(os.path.join(new_root, f"{split}_{str(part).zfill(3)}")))
            % (2 * tar_size)
            == 0
            and len(os.listdir(os.path.join(new_root, f"{split}_{str(part).zfill(3)}")))
            > 0
        ):
            make_tarfile(
                f"{os.path.join(new_root, f'{split}_{str(part).zfill(3)}.tar.gz')}",
                os.path.join(new_root, f"{split}_{str(part).zfill(3)}"),
            )
            shutil.rmtree(os.path.join(new_root, f"{split}_{str(part).zfill(3)}"))
            part += 1

    make_tarfile(
        f"{os.path.join(new_root, f'{split}_{str(part).zfill(3)}.tar.gz')}",
        os.path.join(new_root, f"{split}_{str(part).zfill(3)}"),
    )
    shutil.rmtree(os.path.join(new_root, f"{split}_{str(part).zfill(3)}"))


def halftruth_segments(x):
    x = x.split("/")
    x = [i.split("-")[:-1] for i in x if "F" in i]
    return x


def process_split_halftruth(
    root, new_root, split, temporal_resolution=0.04, tar_size=1000
):
    audio_root = os.path.join(
        root, f"HAD_{split}", "conbine" if split != "test" else "test"
    )
    df = pd.read_csv(
        os.path.join(root, f"HAD_{split}", f"HAD_{split}_label.txt"),
        sep=" ",
        header=None,
    )
    df.columns = ["id", "segments", "label"]

    df["segments"] = df["segments"].apply(lambda x: halftruth_segments(x))
    df = shuffle(df)
    df = df.reset_index()

    new_root = os.path.join(new_root, split)
    if os.path.exists(new_root):
        shutil.rmtree(new_root)
        os.makedirs(new_root)

    part = 0
    for i, row in df.iterrows():
        out_dir = os.path.join(new_root, f"{split}_{str(part).zfill(3)}")
        os.makedirs(out_dir, exist_ok=True)

        audio_file = os.path.join(audio_root, f"{row['id']}.wav")
        audio_frames = wavfile.read(audio_file)[1].size

        shutil.copy(audio_file, out_dir)

        label_df = {
            "audio_path": os.path.join(out_dir, f"{row['id']}.wav"),
            "audio_fake_segments": row["segments"],
            "audio_frames": audio_frames,
        }

        with open(f"{os.path.join(out_dir, row['id'])}.json", "w") as outfile:
            json.dump(label_df, outfile)

        if (
            len(os.listdir(os.path.join(new_root, f"{split}_{str(part).zfill(3)}")))
            % (2 * tar_size)
            == 0
            and len(os.listdir(os.path.join(new_root, f"{split}_{str(part).zfill(3)}")))
            > 0
        ):
            make_tarfile(
                f"{os.path.join(new_root, f'{split}_{str(part).zfill(3)}.tar.gz')}",
                os.path.join(new_root, f"{split}_{str(part).zfill(3)}"),
            )
            shutil.rmtree(os.path.join(new_root, f"{split}_{str(part).zfill(3)}"))
            part += 1

    make_tarfile(
        f"{os.path.join(new_root, f'{split}_{str(part).zfill(3)}.tar.gz')}",
        os.path.join(new_root, f"{split}_{str(part).zfill(3)}"),
    )
    shutil.rmtree(os.path.join(new_root, f"{split}_{str(part).zfill(3)}"))


def create_webdataset(args):
    root = args.data_root
    new_root = root + "_tar"

    if "partialspoof" in root.lower():
        root = os.path.join(root, "database")
        for split in ["train", "dev", "eval"]:
            process_split_partialspoof(root, new_root, split)

    elif "had" in root.lower():
        for split in ["train", "dev", "test"]:
            process_split_halftruth(root, new_root, split)
