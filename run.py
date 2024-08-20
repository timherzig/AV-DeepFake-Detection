from omegaconf import OmegaConf

from src.train import train
from src.eval import eval
from src.util.parser import parse_args
from src.util.utils import get_config


def main(args):
    config = get_config(args.config)
    config.debug = args.debug

    if args.task == "train":
        print(f"Training model")
        train(config, args)
    elif args.task == "evaluate":
        print(f"Evaluating model")
        eval(config, args)

    print(f"Done")


if __name__ == "__main__":
    args = parse_args()
    main(args)
