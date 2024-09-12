from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--task",
        type=str,
        default="train",
        help="train, evaluate, create_webdataset",
        required=True,
    )

    parser.add_argument(
        "--config", type=str, default="config/default.yaml", help="Path to config file"
    )

    parser.add_argument(
        "--data_root", type=str, default="data root path", help="Path to data root"
    )

    parser.add_argument("--debug", action="store_true", help="Debug mode")

    parser.add_argument(
        "--resume", action="store_true", help="Resume training from checkpoint"
    )

    parser.add_argument(
        "--eval_ds", type=str, default="av1m", help="Dataset to evaluate on"
    )

    return parser.parse_args()
