import os
import sys
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from tqdm import tqdm
from torch.nn.functional import softmax

from src.data.data import get_dataloaders
from src.util.metrics import calculate_metrics
from src.util.utils import (
    get_paths,
    get_model_and_checkpoint,
    get_multimodal_model_and_checkpoint,
)

np.set_printoptions(threshold=sys.maxsize)


def transition_eval(config, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, log_dir, model_dir = get_paths(
        config, create_folders=False, evaluate=True, root=args.eval_root
    )

    if "audio-video-lf" in config.model.task:
        model, _ = get_multimodal_model_and_checkpoint(config, args.resume)
    else:
        model, _ = get_model_and_checkpoint(config, model_dir, args.resume)

    config.data.name = args.eval_ds

    _, _, (test_dl, test_len) = get_dataloaders(
        ["test"], args.data_root, config, test=True
    )

    model.to(device)
    model.eval()

    if "audio-video" in config.model.task:
        running_y_true = [np.array([]), np.array([])]
        running_y_pred = [np.array([]), np.array([])]
        running_y_pred_softmax = [np.array([]), np.array([])]
    else:
        running_y_true = np.array([])
        running_y_pred = np.array([])
        running_y_pred_softmax = np.array([])

    with torch.no_grad():
        with tqdm(test_dl, total=math.ceil(test_len / config.train.batch_size)) as pbar:
            for i, batch in enumerate(pbar):
                pbar.set_description(f"Evaluating   ")

                x, y = batch
                if x is None:
                    continue
                if config.model.task == "audio-video":
                    x = (x[0].to(device), x[1].to(device))
                else:
                    x = x.to(device)
                y = y.to(device)

                y_pred = model(x)

                if y_pred.dim() == 3:
                    y_pred1 = softmax(y_pred[:, 0], dim=1)
                    y_pred2 = softmax(y_pred[:, 1], dim=1)
                    y_pred = torch.stack([y_pred1, y_pred2], dim=1)
                else:
                    y_pred = softmax(y_pred, dim=1)
                softmax_predictions = y_pred[:, 0].cpu().detach().numpy()
                y_pred = torch.argmax(y_pred, dim=1).cpu().detach().numpy()
                y = y[:, 0].cpu().detach().numpy().astype(int)
                y_pred = abs(y_pred - 1)

                if "audio-video" in config.model.task:
                    running_y_true[0] = np.concatenate([running_y_true[0], y[0]])
                    running_y_true[1] = np.concatenate([running_y_true[1], y[1]])
                    running_y_pred[0] = np.concatenate([running_y_pred[0], y_pred[0]])
                    running_y_pred[1] = np.concatenate([running_y_pred[1], y_pred[1]])
                    running_y_pred_softmax[0] = np.concatenate(
                        [running_y_pred_softmax[0], softmax_predictions[0]]
                    )
                    running_y_pred_softmax[1] = np.concatenate(
                        [running_y_pred_softmax[1], softmax_predictions[1]]
                    )
                else:
                    running_y_true = np.concatenate([running_y_true, y])
                    running_y_pred = np.concatenate([running_y_pred, y_pred])
                    running_y_pred_softmax = np.concatenate(
                        [running_y_pred_softmax, softmax_predictions]
                    )

    if "audio-video" in config.model.task:
        a_acc, a_f1, a_eer = calculate_metrics(
            running_y_true[0], running_y_pred[0], running_y_pred_softmax[0]
        )
        v_acc, v_f1, v_eer = calculate_metrics(
            running_y_true[1], running_y_pred[1], running_y_pred_softmax[1]
        )

        print(f"Audio Accuracy: {a_acc:.4f}")
        print(f"Audio F1 Score: {a_f1:.4f}")
        print(f"Audio EER: {a_eer:.4f}")

        print(f"Video Accuracy: {v_acc:.4f}")
        print(f"Video F1 Score: {v_f1:.4f}")
        print(f"Video EER: {v_eer:.4f}")

        with open(f"{log_dir}/eval_results_{config.data.name}.txt", "w+") as f:
            f.write(f"Audio Accuracy: {a_acc:.4f}\n")
            f.write(f"Audio F1 Score: {a_f1:.4f}\n")
            f.write(f"Audio EER: {a_eer:.4f}\n")
            f.write(f"Video Accuracy: {v_acc:.4f}\n")
            f.write(f"Video F1 Score: {v_f1:.4f}\n")
            f.write(f"Video EER: {v_eer:.4f}\n")
    else:
        acc, f1, eer = calculate_metrics(
            running_y_true, running_y_pred, running_y_pred_softmax
        )

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

    _, log_dir, model_dir = get_paths(
        config, create_folders=False, evaluate=True, root=args.eval_root
    )
    if "audio-video-lf" in config.model.task:
        model, _ = get_multimodal_model_and_checkpoint(
            config, model_dir, resume=True, eval=True
        )
    else:
        model, _ = get_model_and_checkpoint(config, model_dir, resume=True)

    config.data.name = args.eval_ds

    _, _, (test_dl, test_len) = get_dataloaders(["test"], args.data_root, config)

    model.to(device)
    model.eval()

    print(f"Evaluating with sliding window of size {args.step_size}, batch size {bs}")

    avg_acc = 0
    avg_f1 = 0
    avg_eer = 0
    audio_acc = 0
    video_acc = 0
    audio_f1 = 0
    video_f1 = 0
    audio_eer = 0
    video_eer = 0

    with torch.no_grad():
        with tqdm(test_dl, total=math.ceil(test_len / config.train.batch_size)) as pbar:
            for j, batch in enumerate(pbar):
                pbar.set_description(f"Evaluating   ")
                n_iter = 0
                x, y = batch
                if "audio-video" in config.model.task:
                    x = (x[0].to(device), x[1].to(device))
                    n_iter = x[0].shape[0]
                    predictions = [np.array([], dtype=int), np.array([], dtype=int)]
                    softmaxes = [np.array([], dtype=float), np.array([], dtype=float)]
                else:
                    x = x.to(device)
                    n_iter = x.shape[0]
                    predictions = np.array([], dtype=int)
                    softmaxes = np.array([], dtype=float)
                y = y.to(device)

                for i in range(0, n_iter, bs):
                    if "audio-video" in config.model.task:
                        window = (x[0][i : i + bs, :], x[1][i : i + bs, :])
                    else:
                        window = x[i : i + bs, :]
                    y_pred = model(window)

                    if "audio-video" in config.model.task:
                        a_pred = softmax(y_pred[0], dim=1)
                        softmaxes[0] = np.concatenate(
                            [softmaxes[0], a_pred[:, 0].cpu().detach().numpy()]
                        )
                        a_pred = torch.argmax(a_pred, dim=1).cpu().detach().numpy()
                        v_pred = softmax(y_pred[1], dim=1)
                        softmaxes[1] = np.concatenate(
                            [softmaxes[1], v_pred[:, 0].cpu().detach().numpy()]
                        )
                        v_pred = torch.argmax(v_pred, dim=1).cpu().detach().numpy()

                        predictions[0] = np.concatenate(
                            [predictions[0], a_pred.astype(int)]
                        )
                        predictions[1] = np.concatenate(
                            [predictions[1], v_pred.astype(int)]
                        )

                    else:
                        y_pred = softmax(y_pred, dim=1)
                        softmaxes = np.concatenate(
                            [softmaxes, y_pred[:, 0].cpu().detach().numpy()]
                        )
                        y_pred = torch.argmax(y_pred, dim=1).cpu().detach().numpy()
                        predictions = np.concatenate([predictions, y_pred.astype(int)])

                if "audio-video" in config.model.task:
                    a_y = y[:, 0, 0].cpu().detach().numpy().astype(int)
                    v_y = y[:, 1, 0].cpu().detach().numpy().astype(int)

                    predictions[0] = abs(predictions[0] - 1)
                    predictions[1] = abs(predictions[1] - 1)

                    a_acc, a_f1, a_eer = calculate_metrics(
                        a_y, predictions[0], softmaxes[0]
                    )
                    v_acc, v_f1, v_eer = calculate_metrics(
                        v_y, predictions[1], softmaxes[1]
                    )

                    acc = (a_acc + v_acc) / 2
                    f1 = (a_f1 + v_f1) / 2
                    eer = (a_eer + v_eer) / 2

                    audio_acc += a_acc
                    video_acc += v_acc
                    audio_f1 += a_f1
                    video_f1 += v_f1
                    audio_eer += a_eer
                    video_eer += v_eer
                else:
                    y = y[:, 0].cpu().detach().numpy().astype(int)
                    predictions = abs(predictions - 1)
                    acc, f1, eer = calculate_metrics(y, predictions, softmaxes)

                avg_acc += acc
                avg_f1 += f1
                avg_eer += eer

    avg_acc /= test_len
    avg_f1 /= test_len
    avg_eer /= test_len

    if "audio-video" in config.model.task:
        audio_acc /= test_len
        video_acc /= test_len
        audio_f1 /= test_len
        video_f1 /= test_len
        audio_eer /= test_len
        video_eer /= test_len
        print(
            f" --- Average ---\nAccuracy: {avg_acc:.4f}\nF1 Score: {avg_f1:.4f}\nEER: {avg_eer:.4f}"
        )
        print(
            f" --- Audio ---\nAccuracy: {audio_acc:.4f}\nF1 Score: {audio_f1:.4f}\nEER: {audio_eer:.4f}"
        )
        print(
            f" --- Video ---\nAccuracy: {video_acc:.4f}\nF1 Score: {video_f1:.4f}\nEER: {video_eer:.4f}"
        )
    else:
        print(f"Accuracy: {avg_acc:.4f}")
        print(f"F1 Score: {avg_f1:.4f}")
        print(f"EER: {avg_eer:.4f}")

    with open(
        f"{log_dir}/eval_results_{config.data.name}_sliding_window.txt", "w+"
    ) as f:
        f.write(f"Accuracy: {avg_acc:.4f}\n")
        f.write(f"F1 Score: {avg_f1:.4f}\n")
        f.write(f"EER: {avg_eer:.4f}\n")
        if "audio-video" in config.model.task:
            f.write(f"Audio Accuracy: {audio_acc:.4f}\n")
            f.write(f"Video Accuracy: {video_acc:.4f}\n")
            f.write(f"Audio F1 Score: {audio_f1:.4f}\n")
            f.write(f"Video F1 Score: {video_f1:.4f}\n")
            f.write(f"Average Audio EER: {audio_eer:.4f}\n")
            f.write(f"Average Video EER: {video_eer:.4f}\n")


def single_sliding_window_eval(config, args, bs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.data.sliding_window = True
    config.train.batch_size = 1  # Sliding window only works with batch size 1
    config.data.step_size = args.step_size

    _, log_dir, model_dir = get_paths(
        config, create_folders=False, evaluate=True, root=args.eval_root
    )
    print(f"Loading model from {model_dir}")
    model, _ = get_model_and_checkpoint(config, model_dir, resume=True)

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
                predictions = predictions[
                    config.data.window_size : -config.data.window_size
                ]
                y = y[config.data.window_size : -config.data.window_size]
                softmax_predictions = softmax_predictions[
                    config.data.window_size : -config.data.window_size
                ]

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
                plot_results(f"{log_dir}/{args.eval_ds}", predictions, y, j, config)

                if j == 5:
                    break


def single_transition_eval(config, args, num_samples=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.data.sliding_window = False

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


def plot_results(save_file, predictions, y, j, config):
    time = np.arange(len(predictions))

    cmap = mcolors.LinearSegmentedColormap.from_list("", ["green", "red"])
    plt.figure(figsize=(10, 8))

    plt.bar(
        time,
        predictions,
        color=cmap(predictions),
        label="Predictions",
    )

    x_markers = time[y == 1]
    plt.scatter(
        x_markers,
        [0] * len(x_markers),
        color="red",
        s=200,
        label="Ground truth transitions",
        marker="^",
    )

    plt.xlabel("Time in frames")
    plt.ylabel("Predicted probability")

    plt.legend()
    plt.savefig(f"{save_file}/predictions{j}.png")
    plt.close()
