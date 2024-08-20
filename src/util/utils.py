import os
import torch
import shutil

from functools import partial
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from src.model.model import Model


def get_config(config_path):
    config = OmegaConf.load(config_path)
    config.model.encoder = OmegaConf.load(config.model.encoder)
    config.model.decoder = OmegaConf.load(config.model.decoder)
    return config


def get_paths(config, create_folders=True):
    # Creates the necessary directories and saves the configuration object, returns the paths to the root, log and model directories
    # Parameters
    # ----------
    # config : OmegaConf
    #     Configuration object
    # Returns
    # -------
    # root : str
    #     Root directory
    # log_dir : str
    #     Log directory
    # model_dir : str
    #     Model directory

    root = os.path.join(
        "checkpoints",
        config.data.name,
        config.model.task,
        "decoder_" + config.model.decoder.name,
        "encoder_" + config.model.encoder.name,
    )

    if not os.path.exists(root):
        root = os.path.join(root, "0")
    else:
        if create_folders:
            root = os.path.join(root, str(len(os.listdir(root))))
        else:
            root = os.path.join(root, str(len(os.listdir(root)) - 1))

    if config.debug:
        root = os.path.join("checkpoints", "debug")
        if os.path.exists(root):
            shutil.rmtree(root)

    log_dir = os.path.join(root, "logs")
    os.makedirs(log_dir, exist_ok=True)

    model_dir = os.path.join(root, "models")
    os.makedirs(model_dir, exist_ok=True)

    OmegaConf.save(config, os.path.join(root, "config.yaml"))

    return root, log_dir, model_dir


def get_model_and_checkpoint(config, model_dir, resume=True):
    # Finds and loads the latest checkpoint (if specified) and returns the model along with the checkpoint and the current step
    # Parameters
    # ----------
    # config : OmegaConf
    #     Configuration object
    # model_dir : str
    #    Model directory
    # resume : bool
    #     If True, loads the latest checkpoint
    # Returns
    # -------
    # model : torch.nn.Module
    #     Model object
    # checkpoint : dict
    #     Checkpoint dictionary

    model = Model(config)

    if resume:
        checkpoints = os.listdir(model_dir)
        if len(checkpoints) > 0:
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("_")[1]))
            checkpoint = torch.load(os.path.join(model_dir, checkpoints[-1]))
            model.load_state_dict(checkpoint["model_state_dict"])
            return model, checkpoint

    return model, None


def get_optimizer_and_scheduler(model, config):
    optimizer = None

    if config.train.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    elif config.train.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.train.lr,
            momentum=config.train.momentum,
            weight_decay=config.train.weight_decay,
        )
    else:
        raise ValueError(f"Invalid optimizer: {config.train.optimizer}")

    scheduler = None
    if config.train.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.train.step_size, gamma=config.train.gamma
        )
    elif config.train.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.train.factor,
            patience=config.train.patience,
            verbose=True,
        )
    else:
        raise ValueError(f"Invalid scheduler: {config.train.scheduler}")

    return optimizer, scheduler


def get_critierion(config):
    loss_function = None

    if config.train.loss == "cross_entropy":
        loss_function = torch.nn.CrossEntropyLoss()
    elif config.train.loss == "bce":
        loss_function = torch.nn.BCELoss()
    else:
        raise ValueError(f"Invalid loss function: {config.train.loss}")

    return loss_function


def save_checkpoint(model, model_dir, epoch, val_loss, val_video_acc, val_audio_acc):
    # Saves the model checkpoint
    # Parameters
    # ----------
    # model : torch.nn.Module
    #     Model object
    # model_dir : str
    #     Model directory
    # epoch : int
    #     Current epoch
    # val_loss : float
    #     Validation loss
    # val_video_acc : float
    #     Validation video accuracy
    # val_audio_acc : float
    #     Validation audio accuracy

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
        },
        os.path.join(
            model_dir,
            f"model_{epoch}_{val_loss}_{val_video_acc}_{val_audio_acc}.pt",
        ),
    )


def use_grad(val_results, th, sensitivity):
    # Returns whether to use the gradient based on the early stopping results
    # Parameters
    # ----------
    # val_results : np.array
    #     Array of validation results
    # th : int
    #     Threshold for early stopping
    # Returns
    # -------
    # bool
    #     Whether to use the gradient

    use_grad = sum([i > th for i in val_results]) < sensitivity

    return use_grad
