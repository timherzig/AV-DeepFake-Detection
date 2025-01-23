import os
import math
import torch
import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm

# from torch.utils._triton import has_triton

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.data.data import get_dataloaders
from src.util.utils import get_paths, get_model_and_checkpoint

N_CLUSTERS = 3


def cluster_spoof_types(config, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config.data.return_path = True
    config.data.test_size = 20000

    # if not has_triton():
    #     raise RuntimeError("Triton is not available")

    root, log_dir, model_dir = get_paths(config, create_folders=False, evaluate=True)
    model, _ = get_model_and_checkpoint(config, model_dir, resume=False)

    config.data.name = args.eval_ds

    (_, _), (_, _), (test_dl, test_len) = get_dataloaders(
        ["test"], args.data_root, config, test=True
    )

    model.to(device)
    model.eval()

    encodings = []
    paths = []
    start_end = []

    with torch.no_grad():
        with tqdm(test_dl, total=math.ceil(test_len / config.train.batch_size)) as pbar:
            for i, batch in enumerate(pbar):
                x, y, p, se = batch
                x = x.to(device)
                y = y.to(device)

                if N_CLUSTERS == 3:
                    x = x[y[:, 0] == 1]
                    p = [p[i] for i in range(len(p)) if y[i, 0] == 1]
                    se = [se[i] for i in range(len(se)) if y[i, 0] == 1]

                encoding = model(x, return_encoding=True)
                encodings.append(encoding.cpu().numpy())
                paths.extend(p)
                start_end.extend(se)

    encodings = np.concatenate(encodings, axis=0)
    print(f"final encodings shape: {encodings.shape}")
    print(f"final paths shape: {len(paths)}")

    # encodings = encodings.reshape(
    #     encodings.shape[0], encodings.shape[1] * encodings.shape[2]
    # )

    scaler = StandardScaler()
    encodings = scaler.fit_transform(encodings)

    print(f"encodings shape after scaling: {encodings.shape}")

    pca = PCA(n_components=2)
    encodings = pca.fit_transform(encodings)

    print(f"encodings shape after PCA: {encodings.shape}")

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0).fit(encodings)

    print(f"first 10 paths: {paths[:10]}")

    # get samples from the clusters
    for i in range(N_CLUSTERS):
        cluster_indices = np.where(kmeans.labels_ == i)[0]
        cluster_paths = [paths[i] for i in cluster_indices]
        print(f"Cluster {i} has {len(cluster_paths)} samples")
        with open(f"clustering/cluster_{i}.txt", "w") as f:
            for path in cluster_paths:
                f.write(f"{path}\n")

    plt.scatter(encodings[:, 0], encodings[:, 1], c=kmeans.labels_)
    if not os.path.exists("clustering"):
        os.makedirs("clustering")
    plt.savefig(f"clustering/{args.eval_ds}_kmeans.png")

    return
