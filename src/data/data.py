import torch
from functools import partial

from src.data.av1m.av1m import av1m_get_splits, av1m_collate_fn
from src.data.partialspoof.partialspoof import (
    partialspoof_get_splits,
    partialspoof_collate_fn,
)


def collate_fn(batch, config, test=False):
    if config.data.name == "av1m":
        return av1m_collate_fn(
            batch, config, sliding_window=config.data.sliding_window, test=test
        )
    if config.data.name == "partialspoof":
        return partialspoof_collate_fn(
            batch, config, sliding_window=config.data.sliding_window, test=test
        )

    return None


def get_splits(root, config, train_parts, val_parts, test_parts):
    if config.data.name == "av1m":
        return av1m_get_splits(root, config, train_parts, val_parts, test_parts)
    if config.data.name == "partialspoof":
        return partialspoof_get_splits(root, config, train_parts, val_parts, test_parts)

    return None, None, None


def get_dataloaders(splits, root, config, test=False):
    # Returns the dataloaders for the specified splits along with the lengths of the datasets
    # Parameters
    # ----------
    # splits : list
    #     List of splits to return dataloaders for
    # root : str
    #     Root directory of the dataset
    # config : OmegaConf
    #     Configuration object
    # Returns
    # -------
    # train_dl : torch.utils.data.DataLoader
    #     Dataloader for the training set
    # train_len : int
    #     Length of the training set
    # val_dl : torch.utils.data.DataLoader
    #     Dataloader for the validation set
    # val_len : int
    #     Length of the validation set
    # test_dl : torch.utils.data.DataLoader
    #     Dataloader for the test set
    # test_len : int
    #     Length of the test set

    train_parts = config.data.train_parts
    val_parts = config.data.val_parts
    test_parts = config.data.test_parts

    if config.debug:
        config.data.train_size = 1000
        config.data.val_size = 1000
        config.data.test_size = 1000

    (train, train_len), (val, val_len), (test, test_len) = get_splits(
        root, config, train_parts, val_parts, test_parts
    )

    train_dl, val_dl, test_dl = None, None, None

    c_fn = partial(collate_fn, config=config, test=test)

    if "train" in splits:
        train_dl = torch.utils.data.DataLoader(
            train,
            batch_size=None,
            collate_fn=c_fn,
            num_workers=config.train.num_workers,
        )
    if "val" in splits:
        val_dl = torch.utils.data.DataLoader(
            val,
            batch_size=None,
            collate_fn=c_fn,
            num_workers=config.train.num_workers,
        )
    if "test" in splits:
        test_dl = torch.utils.data.DataLoader(
            test,
            batch_size=None,
            collate_fn=c_fn,
            num_workers=config.train.num_workers,
        )

    return (train_dl, train_len), (val_dl, val_len), (test_dl, test_len)
