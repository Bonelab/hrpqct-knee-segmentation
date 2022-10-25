from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
import numpy as np
import torch
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
        description='2D, 2.5D, or 3D UNet Training Cross-Validation Script',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "data_dirs", type=str, nargs="+", metavar="DIR",
        help="list of directories to pull data from"
    )
    parser.add_argument(
        "--label", "-l", type=str, default='unet', metavar="STR",
        help="base title to use when saving logs and model checkpoints"
    )
    parser.add_argument(
        "--num-workers", "-w", type=int, default=8, metavar="N",
        help="number of CPU workers to use to load data in parallel"
    )
    parser.add_argument(
        "--input-channels", "-ic", type=int, default=1, metavar="N",
        help="Modify this only if you are using a 2.5D model (extra slices on channel axis)."
    )
    parser.add_argument(
        "--output-channels", "-oc", type=int, default=1, metavar="N",
        help="How many classes there are to segment images into."
    )
    parser.add_argument(
        "--model-channels", "-mc", type=int, nargs='+', default=[64, 128, 256, 512], metavar="N",
        help="sequence of filters in U-Net layers"
    )
    parser.add_argument(
        '--is-3d', '-3d', action="store_true", default=False,
        help="set this flag to train a SeGAN with 3D convolutions for segmenting 3D image data"
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
        "--batch-size", "-bs", type=int, default=128, metavar="N",
        help='number of samples per minibatch'
    )
    parser.add_argument(
        "--num-gpus", "-ng", type=int, default=0, metavar="N",
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
    parser.add_argument(
        "--auto-learning-rate", "-alr", action="store_true", default=False,
        help="let pytorch-lightning pick the best learning rate"
    )
    parser.add_argument(
        "--hours-per-fold", "-hpf", type=int, default=4,
        help="maximum time in hours to spend training the model in each fold"
    )

    return parser


def train_unet_2d_cv(args: Namespace) -> None:
    # create datasets
    datasets = []
    for data_dir in args.data_dirs:
        datasets.append(PickledDataset(data_dir))
    dataset = ConcatDataset(datasets)

    # create the fold index lists
    idxs = np.arange(0, len(dataset))
    np.random.shuffle(idxs)
    folds_idxs = np.array_split(idxs, args.folds)

    # dataloader standard kwargs
    dataloader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'pin_memory': True,
        'persistent_workers': True
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
            "spatial_dims": 3 if args.is_3d else 2,
            "in_channels": args.input_channels,  # density
            "out_channels": args.output_channels,  # cort, trab, back
            "channels": args.model_channels,
            "strides": [1 for _ in range(len(args.model_channels) - 1)],
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

        # create callbacks
        early_stopping = EarlyStopping(
            monitor='val_dsc_0',
            mode='max',
            patience=args.early_stopping_patience
        )

        # find learning rate if option enabled
        if args.auto_learning_rate:
            # tune on just one GPU or CPU since you can't do it in parallel
            trainer = Trainer(
                accelerator=("gpu" if args.num_gpus > 0 else "cpu"), devices=1,
                auto_lr_find=True, enable_checkpointing=False
            )
            trainer.tune(task, train_dataloader, val_dataloader)
            args.learning_rate = task.learning_rate

        # create a Trainer and fit the model
        csv_logger.log_hyperparams(args)
        trainer = Trainer(
            accelerator=("gpu" if args.num_gpus > 0 else "cpu"),
            devices=int(np.maximum(args.num_gpus, 1)),
            strategy="ddp_find_unused_parameters_false",
            max_epochs=args.epochs,
            max_time={"hours": args.hours_per_fold},
            log_every_n_steps=args.log_step_interval,
            logger=csv_logger,
            callbacks=[early_stopping]
        )
        trainer.fit(task, train_dataloader, val_dataloader)


def main() -> None:
    # get parameters from command line
    args = create_parser().parse_args()

    training_complete = False

    # start the training attempt loop
    while not training_complete:
        try:
            # attempt to train at current batch size
            train_unet_2d_cv(args)
            # if successful, set the flag to end the while loop
            training_complete = True
        except RuntimeError as err:
            # if we get a runtime error, check if it involves being out of memory
            if 'out of memory' in str(err):
                # if the error was because we're out of memory, then we want to reduce the batch size
                if args.batch_size == 1:
                    # if the batch size is 1, we can't reduce it anymore so give up and raise the error
                    print("CUDA OOM with batch size = 1, reduce model complexity.")
                    raise err
                else:
                    # if the batch size is not 1, divide it by 2 (with integer division), empty the cache, and let
                    # the while loop go around again
                    args.batch_size = args.batch_size // 2
                    torch.cuda.empty_cache()
            else:
                # if the error did not have to do with being out of memory, raise it
                raise err


if __name__ == "__main__":
    main()
