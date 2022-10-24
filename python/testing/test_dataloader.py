from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset
from blpytorchlightning.dataset_components.datasets.PickledDataset import PickledDataset


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description='Script that creates a dataloader and then iterates to check memory consumption of dataset.',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "data_dirs", type=str, nargs="+", metavar="DIR",
        help="list of directories to pull data from"
    )
    parser.add_argument(
        "--data-multiplier", "-dm", type=int, default=100,
        help="how many copies of dataset(s) to use"
    )
    parser.add_argument(
        "--batch-size", "-bs", type=int, default=32, metavar="N",
        help='number of samples per minibatch'
    )
    parser.add_argument(
        "--num-workers", "-w", type=int, default=8, metavar="N",
        help="number of CPU workers to use to load data in parallel"
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=200, metavar="N",
        help="number of epochs to train for"
    )

    return parser


def main():
    # get parameters from command line
    args = create_parser().parse_args()

    # create datasets
    datasets = []
    for data_dir in args.data_dirs:
        for _ in range(args.data_multiplier):
            datasets.append(PickledDataset(data_dir))
    dataset = ConcatDataset(datasets)

    # dataloader standard kwargs
    dataloader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'pin_memory': True,
        'persistent_workers': True
    }

    dataloader = DataLoader(dataset, **dataloader_kwargs)

    for epoch in range(args.epochs):
        for i, item in enumerate(dataloader):
            mem = int(psutil.Process().memory_info().rss / (1024 * 1024))
            print(f"epoch {epoch:4d}/{args.epochs-1:4d}, batch {i:4d}/{len(dataloader)-1:4d}, memory: {mem:8d}")


if __name__ == "__main__":
    main()
