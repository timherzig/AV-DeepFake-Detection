from src.eval import (
    transition_eval,
    sliding_window_eval,
    single_sliding_window_eval,
    single_transition_eval,
)
from src.train import train
from src.util.utils import get_config
from src.util.parser import parse_args
from src.util.misc.model_stats import log_model_stats
from src.util.misc.iterate_dataset import iterate_dataset
from src.util.misc.create_webdataset import create_webdataset
from src.util.misc.cluster_spoof_types import cluster_spoof_types


def main(args):
    config = get_config(args.config)
    config.debug = args.debug
    # config.data.overlap_add = args.overlap_add
    config.data.eval_overlap_add = args.eval_overlap_add
    config.data.return_path = False

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
    elif args.task == "cluster":
        print(f"Clustering spoof types")
        cluster_spoof_types(config, args)
    elif args.task == "log_model_stats":
        print(f"Logging model stats")
        log_model_stats(config, args)

    print(f"Done")


if __name__ == "__main__":
    args = parse_args()
    main(args)
