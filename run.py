from src.eval import (
    transition_eval,
    sliding_window_eval,
    single_sliding_window_eval,
    single_transition_eval,
)
from src.train import train
from src.util.utils import get_config
from src.util.parser import parse_args
from src.util.misc.iterate_dataset import iterate_dataset
from src.util.misc.create_webdataset import create_webdataset


def main(args):
    config = get_config(args.config)
    config.debug = args.debug

    if args.debug:
        config.train.batch_size = 4

    if args.task == "train":
        print(f"Training model")
        train(config, args)
    elif args.task == "evaluate":
        print(f"Evaluating model separate centered windows")
        transition_eval(config, args)
    elif args.task == "evaluate_sw":
        print(f"Evaluating model sliding window")
        sliding_window_eval(config, args, config.train.batch_size)
    elif args.task == "evaluate_sw_single":
        config.train.batch_size = 4
        print(f"Evaluating model sliding window")
        single_sliding_window_eval(config, args, config.train.batch_size)
        print(f"----------------------------------------------------------------------")
        # single_transition_eval(config, args)
    elif args.task == "create_webdataset":
        print(f"Preparing webdataset")
        create_webdataset(args)
    elif args.task == "iterate_dataset":
        print(f"Iterating dataset")
        iterate_dataset(args)

    print(f"Done")


if __name__ == "__main__":
    args = parse_args()
    main(args)
