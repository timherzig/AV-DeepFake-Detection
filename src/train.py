import math
import torch
import numpy as np

from tqdm import tqdm
from functools import partial
from torch.nn.functional import softmax
from torch.utils._triton import has_triton
from torch.utils.tensorboard import SummaryWriter

from src.util.utils import (
    get_paths,
    get_model_and_checkpoint,
    get_optimizer_and_scheduler,
    get_critierion,
    save_checkpoint,
    use_grad,
)

from src.data.data import get_dataloaders
from src.util.logger import log_train_step, log_val_step
from src.util.metrics import calculate_metrics


def train(config, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not has_triton():
        raise RuntimeError("Triton is not available")

    root, log_dir, model_dir = get_paths(config, create_folders=not args.resume)

    writer = SummaryWriter(log_dir=log_dir)

    (train_dl, train_len), (val_dl, val_len), _ = get_dataloaders(
        ["train", "val"], args.data_root, config, test=False
    )

    model, checkpoint = get_model_and_checkpoint(config, model_dir, args.resume)
    optimizer, scheduler = get_optimizer_and_scheduler(model, config)
    criterion = get_critierion(config)

    model.to(device)

    best_val_loss = float("inf")

    print(f"Starting training on {device}")

    for epoch in range(1, config.train.num_epochs + 1):
        train_loss = train_epoch(
            model,
            criterion,
            optimizer,
            scheduler,
            train_dl,
            train_len,
            epoch,
            config,
            writer,
            device,
        )

        print(f"Epoch {epoch} - Train Loss: {train_loss}")
        save_checkpoint(
            model,
            model_dir,
            epoch,
            train_loss,
            0,
        )

        val_loss, val_acc, val_f1, val_eer = val_epoch(
            model,
            criterion,
            val_dl,
            val_len,
            epoch,
            config,
            writer,
            device,
        )

        print(
            f"Epoch {epoch} - Val Loss: {val_loss} - Val Acc: {val_acc} - Val F1: {val_f1} - Val EER: {val_eer}"
        )

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model,
                model_dir,
                epoch,
                val_loss,
                val_f1,
            )


def train_epoch(
    model,
    criterion,
    optimizer,
    scheduler,
    train_dl,
    train_len,
    epoch,
    config,
    writer,
    device,
):
    model.train()
    logger = partial(log_train_step, config=config, epoch=epoch, writer=writer)
    running_loss = 0.0

    with tqdm(
        train_dl, total=math.ceil(train_len / config.train.batch_size), unit="b"
    ) as pbar:
        for i, batch in enumerate(pbar):
            pbar.set_description(f"Epoch {epoch} - Train")

            x, y = batch

            if x is None and y is None:
                continue

            if config.model.task == "audio-video":
                x = (x[0].to(device), x[1].to(device))
            else:
                x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_pred = model(x)

            if "logits" not in config.train.loss:
                y_pred = softmax(y_pred, dim=1)

            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_loss = logger(i, running_loss, pbar, train_len)

    return running_loss


def val_epoch(
    model,
    criterion,
    val_dl,
    val_len,
    epoch,
    config,
    writer,
    device,
):
    model.eval()
    logger = partial(log_val_step, config=config, epoch=epoch, writer=writer)
    iters = 0
    running_loss = 0.0

    if config.model.task == "audio-video":
        running_y_true = [np.array([]), np.array([])]
        running_y_pred = [np.array([]), np.array([])]
        running_softmax = [np.array([]), np.array([])]
    else:
        running_y_true = np.array([])
        running_y_pred = np.array([])
        running_softmax = np.array([])

    with tqdm(
        val_dl, total=math.ceil(val_len / config.train.batch_size), unit="b"
    ) as pbar:
        for i, batch in enumerate(pbar):
            pbar.set_description(f"Epoch {epoch} - Val  ")
            iters += 1

            x, y = batch

            if x is None and y is None:
                continue

            if config.model.task == "audio-video":
                x = (x[0].to(device), x[1].to(device))
            else:
                x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
                y_pred = model(x)

                if "logits" not in config.train.loss:
                    y_pred = softmax(y_pred, dim=1)
                loss = criterion(y_pred, y)

            y_pred = softmax(y_pred, dim=1)
            y_softmax = y_pred[:, 0].cpu().detach().numpy()
            y_pred = torch.argmax(y_pred, dim=1).cpu().detach().numpy()
            y = torch.argmax(y, dim=1).cpu().detach().numpy()

            if config.model.task == "audio-video":
                running_y_true[0] = np.concatenate([running_y_true[0], y[0]])
                running_y_true[1] = np.concatenate([running_y_true[1], y[1]])
                running_y_pred[0] = np.concatenate([running_y_pred[0], y_pred[0]])
                running_y_pred[1] = np.concatenate([running_y_pred[1], y_pred[1]])
                running_softmax[0] = np.concatenate([running_softmax[0], y_softmax[0]])
                running_softmax[1] = np.concatenate([running_softmax[1], y_softmax[1]])
            else:
                running_y_true = np.concatenate([running_y_true, y])
                running_y_pred = np.concatenate([running_y_pred, y_pred])
                running_softmax = np.concatenate([running_softmax, y_softmax])

            running_loss += loss.item()
            logger(
                i,
                running_loss,
                pbar,
                val_len,
            )

    running_loss /= iters

    if config.model.task == "audio-video":
        a_acc, a_f1, a_eer = calculate_metrics(
            running_y_true[0], running_y_pred[0], running_softmax[0]
        )
        v_acc, v_f1, v_eer = calculate_metrics(
            running_y_true[1], running_y_pred[1], running_softmax[1]
        )

        acc = (a_acc + v_acc) / 2
        f1 = (a_f1 + v_f1) / 2
        eer = (a_eer + v_eer) / 2
    else:
        acc, f1, eer = calculate_metrics(
            running_y_true, running_y_pred, running_softmax
        )

    return running_loss, acc, f1, eer
