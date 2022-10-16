from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, ConcatDataset, Subset
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from blpytorchlightning.tasks.SegmentationTask import SegmentationTask
from blpytorchlightning.dataset_components.datasets.PickledDataset import PickledDataset
from monai.networks.nets.unet import UNet


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description='2D UNet Training Cross-Validation Script',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "data_dirs", type=str, nargs="+", metavar="DIR",
        help="list of directories to pull data from"
    )
    parser.add_argument(
        "--label", "-l", type=str, default='unet2d', metavar="STR",
        help="base title to use when saving logs and model checkpoints"
    )
    parser.add_argument(
        "--num-workers", "-w", type=int, default=8, metavar="N",
        help="number of CPU workers to use to load data in parallel"
    )
    parser.add_argument(
        "--channels", "-c", type=int, nargs='+', default=[64, 128, 256, 512, 1024], metavar="N",
        help="sequence of filters in U-Net layers"
    )
    parser.add_argument(
        "--dropout", "-d", type=float, default=0.1, metavar="D",
        help="dropout probability"
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=200, metavar="N",
        help="number of epochs to train for"
    )
    parser.add_argument(
        "--learning-rate", "-lr", type=float, default=0.001, metavar="LR",
        help="learning rate for the optimizer"
    )
    parser.add_argument(
        "--batch-size", "-bs", type=int, default=32, metavar="N",
        help='number of samples per minibatch'
    )
    parser.add_argument(
        "--num_gpus", "-ng", type=int, default=0, metavar="N",
        help="number of GPUs to use"
    )
    parser.add_argument(
        "--version", "-v", type=int, default=None, metavar="N",
        help="version number for logging"
    )
    parser.add_argument(
        "--folds", "-f", type=int, default=5, metavar="N",
        help="number of folds to use in cross-validation"
    )
    parser.add_argument(
        "--log-step-interval", "-lsi", type=int, default=20, metavar="N",
        help="log metrics every N training/validation steps"
    )
    parser.add_argument(
        '--early-stopping-patience', '-esp', type=int, default=40, metavar='N',
        help='number of epochs to train for'
    )

    return parser


def main() -> None:
    # get parameters from command line
    args = create_parser().parse_args()

    # create datasets
    datasets = []
    for data_dir in args.data_dirs:
        datasets.append(PickledDataset(data_dir))
    dataset = ConcatDataset(datasets)

    # create the fold index lists
    idxs = np.arange(0, len(dataset))
    np.random.shuffle(idxs)
    folds_idxs = np.array_split(idxs, args.folds)

    print(folds_idxs)

    # dataloader standard kwargs
    dataloader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'pin_memory': True
    }

    # start the cross-validation loop
    for f in range(args.folds):

        # create dataloaders
        train_dataloader = DataLoader(
            Subset(
                dataset,
                np.concatenate([folds_idxs[i] for i in range(args.folds) if i is not f])
            ),
            shuffle=True, **dataloader_kwargs
        )
        val_dataloader = DataLoader(
            Subset(dataset, folds_idxs[f]),
            **dataloader_kwargs
        )

        # create the model
        model_kwargs = {
            "spatial_dims": 2,  # 2D segmentation
            "in_channels": 1,  # density
            "out_channels": 3,  # cort, trab, back
            "channels": args.channels,
            "strides": [1 for _ in range(len(args.channels) - 1)],
            "dropout": args.dropout
        }
        model = UNet(**model_kwargs)
        model.float()

        # create the task
        task = SegmentationTask(
            model, CrossEntropyLoss(),
            learning_rate=args.learning_rate
        )

        # create loggers
        csv_logger = CSVLogger(
            './logs',
            name=args.label,
            version=f"{args.version}_f{f}"
        )
        csv_logger.log_hyperparams(args)

        # create callbacks
        early_stopping = EarlyStopping(
            monitor='val_dsc_0',
            mode='max',
            patience=args.early_stopping_patience
        )

        # create a Trainer
        trainer = Trainer(
            gpus=args.num_gpus,
            max_epochs=args.epochs,
            log_every_n_steps=args.log_step_interval,
            logger=csv_logger,
            callbacks=[early_stopping]
        )
        trainer.fit(task, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
