import math
import torch
import numpy as np

from tqdm import tqdm
from functools import partial
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


def train(config, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not has_triton():
        raise RuntimeError("Triton is not available")

    writer = SummaryWriter(log_dir=log_dir)

    root, log_dir, model_dir = get_paths(config, create_folders=not args.resume)

    (train_dl, train_len), (val_dl, val_len), _ = get_dataloaders(
        ["train", "val"], args.data_root, config
    )

    model, checkpoint = get_model_and_checkpoint(config, model_dir, args.resume)
    optimizer, scheduler = get_optimizer_and_scheduler(model, config)
    criterion = get_critierion(config)

    model.to(device)

    best_val_loss = float("inf")

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

        val_loss = val_epoch(
            model,
            criterion,
            val_dl,
            val_len,
            epoch,
            writer,
            device,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_loss,
                model_dir,
                config,
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
        train_dl, total=math.ceil(train_len / train_dl.batch_size), unit="b"
    ) as pbar:
        for i, batch in enumerate(pbar):
            pbar.set_description(f"Epoch {epoch} - Train")

            (x, y), info = batch

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            y_pred = model(x)
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
    writer,
    device,
):
    model.eval()
    logger = partial(log_val_step, writer=writer)
    running_loss = 0.0

    with tqdm(val_dl, total=math.ceil(val_len / val_dl.batch_size), unit="b") as pbar:
        for i, batch in enumerate(pbar):
            pbar.set_description(f"Epoch {epoch} - Val")

            (x, y), info = batch

            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
                y_pred = model(x)
                loss = criterion(y_pred, y)

                running_loss += loss.item()

                running_loss = logger(loss.item(), i, epoch)

    return running_loss
