import os
import json
import webdataset as wds


def sort_key(x):
    return int(x.split(".")[0].split("_")[-1])


def json_decoder(data):
    data = json.loads(data)
    return data


def iterate_split(root, args):
    print(f"Processing {root}")
    tar_list = os.listdir(root)
    tar_list = sorted(tar_list, key=sort_key)

    for i, tar in enumerate(tar_list):
        if not tar.endswith(".tar.gz"):
            print(f"Skipping {tar} <--------------------------")
            continue

        print(f"Extracting {tar}, {i+1} of {len(tar_list)}")

        dataset = (
            wds.WebDataset(os.path.join(root, tar))
            .decode(
                wds.torch_video,
                wds.handle_extension("json", json_decoder),
            )
            .to_tuple("mp4", "json")
            .batched(64)
        )

        for i, (video, meta) in enumerate(dataset):
            if i % 1000 == 0:
                print(f"Processed {i} samples")

    return


def iterate_dataset(args):
    root = args.data_root

    if os.path.exists(os.path.join(root, "train")):
        iterate_split(os.path.join(root, "train"), args)

    # AV1M case
    if os.path.exists(os.path.join(root, "val")):
        iterate_split(os.path.join(root, "val"), args)

    # PartialSpoof case
    if os.path.exists(os.path.join(root, "dev")):
        iterate_split(os.path.join(root, "dev"), args)

    if os.path.exists(os.path.join(root, "test")):
        iterate_split(os.path.join(root, "test"), args)
