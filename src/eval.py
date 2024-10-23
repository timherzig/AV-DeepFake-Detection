import os
import sys
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
from torch.nn.functional import softmax
from torch.utils._triton import has_triton

from src.data.data import get_dataloaders
from src.util.metrics import calculate_metrics
from src.util.utils import get_paths, get_model_and_checkpoint

np.set_printoptions(threshold=sys.maxsize)


def transition_eval(config, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not has_triton():
        raise RuntimeError("Triton is not available")

    _, log_dir, model_dir = get_paths(
        config, create_folders=False, evaluate=True, root=args.eval_root
    )
    model, _ = get_model_and_checkpoint(config, model_dir, True)

    config.data.name = args.eval_ds

    _, _, (test_dl, test_len) = get_dataloaders(
        ["test"], args.data_root, config, test=True
    )

    model.to(device)
    model.eval()

    running_y_true = np.array([])
    running_y_pred = np.array([])

    with torch.no_grad():
        with tqdm(test_dl, total=math.ceil(test_len / config.train.batch_size)) as pbar:
            for i, batch in enumerate(pbar):
                pbar.set_description(f"Evaluating   ")

                x, y = batch
                x = x.to(device)
                y = y.to(device)

                y_pred = model(x)

                y_pred = softmax(y_pred, dim=1)
                y_pred = torch.argmax(y_pred, dim=1).cpu().detach().numpy()
                y = y[:, 0].cpu().detach().numpy().astype(int)
                y_pred = abs(y_pred - 1)

                running_y_true = np.concatenate([running_y_true, y])
                running_y_pred = np.concatenate([running_y_pred, y_pred])

    # tmp fix to avoid all neg predictions
    # running_y_true = np.concatenate([running_y_true, np.array([1]), np.array([0])])
    # running_y_pred = np.concatenate([running_y_pred, np.array([1]), np.array([0])])

    acc, f1, eer = calculate_metrics(running_y_true, running_y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"EER: {eer:.4f}")

    with open(f"{log_dir}/eval_results_{config.data.name}.txt", "w+") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"EER: {eer:.4f}\n")


def sliding_window_eval(config, args, bs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.data.sliding_window = True
    config.train.batch_size = 1  # Sliding window only works with batch size 1
    config.data.step_size = args.step_size

    if not has_triton():
        raise RuntimeError("Triton is not available")

    _, log_dir, model_dir = get_paths(
        config, create_folders=False, evaluate=True, root=args.eval_root
    )
    model, _ = get_model_and_checkpoint(config, model_dir, True)

    config.data.name = args.eval_ds

    _, _, (test_dl, test_len) = get_dataloaders(["test"], args.data_root, config)

    model.to(device)
    model.eval()

    print(f"Evaluating with sliding window of size {args.step_size}, batch size {bs}")

    avg_acc = 0
    avg_f1 = 0
    avg_eer = 0

    with torch.no_grad():
        with tqdm(test_dl, total=math.ceil(test_len / config.train.batch_size)) as pbar:
            for j, batch in enumerate(pbar):
                pbar.set_description(f"Evaluating   ")

                x, y = batch
                x = x.to(device)
                y = y.to(device)

                predictions = np.array([], dtype=int)

                for i in range(0, x.shape[0], bs):
                    window = x[i : i + bs, :]
                    y_pred = model(window)

                    y_pred = softmax(y_pred, dim=1)

                    y_pred = torch.argmax(y_pred, dim=1).cpu().detach().numpy()

                    predictions = np.concatenate([predictions, y_pred.astype(int)])

                y = y[:, 0].cpu().detach().numpy().astype(int)
                predictions = abs(predictions - 1)

                acc, f1, eer = calculate_metrics(y, predictions)

                avg_acc += acc
                avg_f1 += f1
                avg_eer += eer

    avg_acc /= test_len
    avg_f1 /= test_len
    avg_eer /= test_len

    print(f"Accuracy: {avg_acc:.4f}")
    print(f"F1 Score: {avg_f1:.4f}")
    print(f"EER: {avg_eer:.4f}")

    with open(
        f"{log_dir}/eval_results_{config.data.name}_sliding_window.txt", "w+"
    ) as f:
        f.write(f"Accuracy: {avg_acc:.4f}\n")
        f.write(f"F1 Score: {avg_f1:.4f}\n")
        f.write(f"EER: {avg_eer:.4f}\n")


def single_sliding_window_eval(config, args, bs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.data.sliding_window = True
    config.train.batch_size = 1  # Sliding window only works with batch size 1
    config.data.step_size = args.step_size

    if not has_triton():
        raise RuntimeError("Triton is not available")

    _, log_dir, model_dir = get_paths(
        config, create_folders=False, evaluate=True, root=args.eval_root
    )
    model, _ = get_model_and_checkpoint(config, model_dir, True)

    config.data.name = args.eval_ds

    _, _, (test_dl, test_len) = get_dataloaders(["test"], args.data_root, config)

    model.to(device)
    model.eval()

    print(
        f"Evaluating a single instance with sliding window step size {args.step_size}, batch size {bs}"
    )

    with torch.no_grad():
        with tqdm(test_dl, total=math.ceil(test_len / config.train.batch_size)) as pbar:
            for j, batch in enumerate(pbar):
                x, y = batch
                x = x.to(device)
                y = y.to(device)

                softmax_predictions = np.array([], dtype=float)
                predictions = np.array([], dtype=int)

                for i in range(0, x.shape[0], bs):
                    window = x[i : i + bs, :]
                    y_pred = model(window)

                    y_pred_softmax = softmax(y_pred, dim=1)
                    softmax_predictions = np.concatenate(
                        [
                            softmax_predictions,
                            y_pred_softmax[:, 0].cpu().detach().numpy(),
                        ]
                    )

                    y_pred = torch.argmax(y_pred_softmax, dim=1).cpu().detach().numpy()
                    predictions = np.concatenate([predictions, y_pred.astype(int)])

                predictions = abs(predictions - 1)
                y = y[:, 0].cpu().detach().numpy().astype(int)

                print(
                    f"Difference between predictions and true: {np.sum(predictions != y)} of {len(y)}"
                )

                print(
                    f"Number of transitions found where y = 1: {np.sum((predictions == 1) & (y == 1))} of {np.sum(y == 1)}"
                )
                print(
                    f"Number of non-transitions found where y = 0: {np.sum((predictions == 0) & (y == 0))} of {np.sum(y == 0)}"
                )
                print(
                    f"Number of false positives: {np.sum((predictions == 1) & (y == 0))}"
                )
                print(
                    f"Number of false negatives: {np.sum((predictions == 0) & (y == 1))}"
                )

                os.makedirs(f"{log_dir}/{args.eval_ds}", exist_ok=True)
                plot_results(f"{log_dir}/{args.eval_ds}", predictions, y, j)

                acc, f1, eer = calculate_metrics(y, predictions)
                print(f"Accuracy: {acc:.4f}")
                print(f"F1 Score: {f1:.4f}")
                print(f"EER: {eer:.4f}")

                if j == 5:
                    break


def single_transition_eval(config, args, num_samples=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.data.sliding_window = False

    if not has_triton():
        raise RuntimeError("Triton is not available")

    _, log_dir, model_dir = get_paths(
        config, create_folders=False, evaluate=True, root=args.eval_root
    )
    model, _ = get_model_and_checkpoint(config, model_dir, True)

    config.data.name = args.eval_ds

    _, _, (test_dl, test_len) = get_dataloaders(
        ["test"], args.data_root, config, test=True
    )

    model.to(device)
    model.eval()

    running_y_true = np.array([])
    running_y_pred = np.array([])

    with torch.no_grad():
        with tqdm(test_dl, total=math.ceil(test_len / config.train.batch_size)) as pbar:
            for i, batch in enumerate(pbar):
                pbar.set_description(f"Evaluating   ")

                x, y = batch
                x = x.to(device)
                y = y.to(device)

                y_pred = model(x)

                y_pred = softmax(y_pred, dim=1)
                y_pred = torch.argmax(y_pred, dim=1).cpu().detach().numpy()
                y = y[:, 0].cpu().detach().numpy().astype(int)
                y_pred = abs(y_pred - 1)

                running_y_true = np.concatenate([running_y_true, y])
                running_y_pred = np.concatenate([running_y_pred, y_pred])

                if i == num_samples:
                    break

    y = running_y_true
    predictions = running_y_pred

    acc, f1, eer = calculate_metrics(y, predictions)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"EER: {eer:.4f}")

    print(
        f"Difference between predictions and true: {np.sum(predictions != y)} of {len(y)}"
    )
    print(
        f"Number of transitions found where y = 1: {np.sum((predictions == 1) & (y == 1))} of {np.sum(y == 1)}"
    )
    print(
        f"Number of non-transitions found where y = 0: {np.sum((predictions == 0) & (y == 0))} of {np.sum(y == 0)}"
    )
    print(f"Number of false positives: {np.sum((predictions == 1) & (y == 0))}")
    print(f"Number of false negatives: {np.sum((predictions == 0) & (y == 1))}")


def plot_results(save_file, predictions, y, j):
    time = np.arange(len(predictions))

    cmap = mcolors.LinearSegmentedColormap.from_list("", ["green", "red"])
    plt.figure(figsize=(10, 5))

    plt.bar(
        time,
        y,
        color="blue",
        # linewidth=1,
        label="Ground truth",
    )
    plt.bar(
        time,
        predictions,
        # linewidth=1,
        # linestyle="--",
        color=cmap(predictions),
        label="Predictions",
        alpha=0.5,
    )

    plt.legend()
    plt.savefig(f"{save_file}/predictions{j}.png")
    plt.close()
