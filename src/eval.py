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

    root, log_dir, model_dir = get_paths(config, create_folders=False, evaluate=True)

    _, _, (test_dl, test_len) = get_dataloaders(
        ["test"], args.data_root, config, test=True
    )

    model, checkpoint = get_model_and_checkpoint(config, model_dir, True)

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

    with open(f"{log_dir}/eval_results_{config.data.name}.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"EER: {eer:.4f}\n")


def sliding_window_eval(config, args, bs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.data.sliding_window = True
    config.data.batch_size = 1  # Sliding window only works with batch size 1

    if not has_triton():
        raise RuntimeError("Triton is not available")

    root, log_dir, model_dir = get_paths(config, create_folders=False, evaluate=True)

    _, _, (test_dl, test_len) = get_dataloaders(["test"], args.data_root, config)

    model, checkpoint = get_model_and_checkpoint(config, model_dir, True)

    model.to(device)
    model.eval()

    with torch.no_grad():
        with tqdm(test_dl, total=math.ceil(test_len / config.train.batch_size)) as pbar:
            for i, batch in enumerate(pbar):
                pbar.set_description(f"Evaluating   ")

                x, y = batch
                x = x.to(device)
                y = y.to(device)

                predictions = np.array([])

                for i in range(0, x.shape[0], bs):
                    window = x[i : i + bs, :]
                    y_pred = model(window)

                    y_pred = softmax(y_pred, dim=1)

                    y_pred = torch.argmax(y_pred, dim=1).cpu().detach().numpy()
                    predictions = np.concatenate([predictions, y_pred])

                y = torch.argmax(y, dim=1).cpu().detach().numpy()

                acc, f1, eer = calculate_metrics(y, predictions)
                print(f"Accuracy: {acc:.4f}")
                print(f"F1 Score: {f1:.4f}")
                print(f"EER: {eer:.4f}")
