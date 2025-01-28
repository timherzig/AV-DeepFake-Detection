import os
import json
import webdataset as wds


def sort_key(x):
    return int(x.split(".")[0].split("_")[-1])


def json_decoder(data):
    data = json.loads(data)
    return data


def iterate_split(root, args, media):
    tar_list = os.listdir(root)
    tar_list = sorted(tar_list, key=sort_key)

    real = 0
    fake_audio = 0
    # fake_video = 0
    # fake_audio_fake_video = 0

    # yourtts = 0
    # vits = 0
    # no_tts = 0

    for i, tar in enumerate(tar_list):
        if not tar.endswith(".tar.gz"):
            print(f"Skipping {tar} <--------------------------")
            continue

        print(f"Extracting {tar}, {i+1} of {len(tar_list)}")
        # print(f"Extracting {tar}, {i+1} of {len(tar_list)}")

        dataset = (
            wds.WebDataset(os.path.join(root, tar))
            .decode(
                wds.torch_video if media == "mp4" else wds.torch_audio,
                wds.handle_extension("json", json_decoder),
            )
            .to_tuple(media, "json")
            .batched(64)
        )

        for i, (video, meta) in enumerate(dataset):
            # for m in meta:
            #     if len(m["audio_fake_segments"]) == 0:
            #         real += 1
            #     else:
            #         fake_audio += 1
            video_len = len(video)

    # with open(f"{root + '_stats.txt'}", "w") as f:
    #     f.write(
    #         f"Real: {real}, Fake Audio only: {fake_audio}"  # , Fake Video only: {fake_video}, Fake Audio and Video {fake_audio_fake_video}, YourTTS: {yourtts}, VITS: {vits}, No TTS: {no_tts}"
    #     )

    # print(
    #     f"Real: {real}, Fake Audio only: {fake_audio}"  # , Fake Video only: {fake_video}, Fake Audio and Video {fake_audio_fake_video}, YourTTS: {yourtts}, VITS: {vits}, No TTS: {no_tts}"
    # )

    return


def iterate_dataset(args):
    root = args.data_root
    media = "mp4" if args.eval_ds == "av1m" else "wav"

    # if os.path.exists(os.path.join(root, "train")):
    #     print(f"Processing Train - {root}")
    #     iterate_split(os.path.join(root, "train"), args, media)

    # AV1M case
    if os.path.exists(os.path.join(root, "val")):
        print(f"Processing Val - {root}")
        iterate_split(os.path.join(root, "val"), args, media)

    # PartialSpoof case
    # if os.path.exists(os.path.join(root, "dev")):
    #     iterate_split(os.path.join(root, "dev"), args, media)

    # AV1M case
    # if os.path.exists(os.path.join(root, "test")):
    #     print(f"Processing Test - {root}")
    #     iterate_split(os.path.join(root, "test"), args, media)

    # PartialSpoof case
    # if os.path.exists(os.path.join(root, "eval")):
    #     iterate_split(os.path.join(root, "eval"), args, media)
