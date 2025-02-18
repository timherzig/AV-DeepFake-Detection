import os
import torch

from src.util.utils import (
    get_paths,
    get_model_and_checkpoint,
    get_multimodal_model_and_checkpoint,
)


def log_model_stats(config, args):
    """
    Logs stats about the model, such as number of parameters and memory usage.

    Args:
        config: Config object containing the model and data configuration.
        args: Command line arguments.

    Returns:
        None
    """

    log_file = os.path.join(args.eval_root, "notes.txt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, model_dir = get_paths(
        config, create_folders=False, evaluate=True, root=args.eval_root
    )

    if "audio-video-lf" in config.model.task:
        model, checkpoint = get_multimodal_model_and_checkpoint(config, args.resume)
    else:
        model, checkpoint = get_model_and_checkpoint(config, model_dir, args.resume)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    with open(log_file, "a") as f:
        f.write(f"\nNumber of parameters: {num_params}\n")
        f.write(f"Number of trainable parameters: {num_trainable_params}\n")

    return None
