import os
import math
import torch
import numpy as np
from tqdm import tqdm
from torch.nn.functional import softmax
from torch.utils._triton import has_triton

from src.data.data import get_dataloaders
from src.util.metrics import calculate_metrics
from src.util.utils import get_paths, get_model_and_checkpoint


def transition_eval(config, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not has_triton():
        raise RuntimeError("Triton is not available")

    _, log_dir, model_dir = get_paths(config, create_folders=False, evaluate=True)
    model, checkpoint = get_model_and_checkpoint(config, model_dir, True)

    config.data.name = args.eval_ds
    root, _, _ = get_paths(config, create_folders=False, evaluate=True)

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
                y = torch.argmax(y, dim=1).cpu().detach().numpy()

                running_y_true = np.concatenate([running_y_true, y])
                running_y_pred = np.concatenate([running_y_pred, y_pred])

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
    config.data.batch_size = 1  # Sliding window only works with batch size 1
    config.data.step_size = args.step_size

    if not has_triton():
        raise RuntimeError("Triton is not available")

    _, log_dir, model_dir = get_paths(config, create_folders=False, evaluate=True)
    model, checkpoint = get_model_and_checkpoint(config, model_dir, True)

    config.data.name = args.eval_ds
    root, _, _ = get_paths(config, create_folders=False, evaluate=True)

    _, _, (test_dl, test_len) = get_dataloaders(["test"], args.data_root, config)

    model.to(device)
    model.eval()

    avg_acc = 0
    avg_f1 = 0
    avg_eer = 0

    with torch.no_grad():
        with tqdm(test_dl, total=math.ceil(test_len / config.train.batch_size)) as pbar:
            for i, batch in enumerate(pbar):
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

                y = torch.argmax(y, dim=1).cpu().detach().numpy()

                y = abs(y - 1)
                predictions = abs(predictions - 1)

                cleaned_predictions = np.zeros_like(predictions)

                # Series of 1s are converted to 1
                continue_i = 0
                for i, p in enumerate(predictions):
                    if i < continue_i:
                        continue
                    if p == 1:
                        start = i
                        for j, p2 in enumerate(predictions[i:]):
                            if p2 == 0:
                                end = max(j - 1, 0)
                                continue_i = i + j
                                break

                        cleaned_predictions[start + end // 2] = 1

                # print(f"GT Indicies: {np.where(y == 1)[0]}")
                # print(f"Pred Indicies: {np.where(cleaned_predictions == 1)[0]}")

                acc, f1, eer = calculate_metrics(y, cleaned_predictions)

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
