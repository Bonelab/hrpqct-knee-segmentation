from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
import numpy as np
import torch
from torch.nn import L1Loss
from torch.utils.data import DataLoader, ConcatDataset, Subset
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from blpytorchlightning.tasks.SeGANTask import SeGANTask
from blpytorchlightning.dataset_components.datasets.PickledDataset import PickledDataset
from blpytorchlightning.models.SeGAN import get_segmentor_and_discriminators
from glob import glob
from shutil import rmtree


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description='2D, 2.5D, or 3D SeGAN Training Cross-Validation Script',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "data_dirs", type=str, nargs="+", metavar="DIR",
        help="list of directories to pull data from"
    )
    parser.add_argument(
        "--label", "-l", type=str, default='segan', metavar="STR",
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
        '--channels-per-group', '-cpg', type=int, default=16, metavar='N',
        help='channels per group in GroupNorm'
    )
    parser.add_argument(
        '--upsample-mode', '-us', type=str, default="bilinear", metavar='STR',
        help='method for upsampling in decoder'
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
        "--epochs", "-e", type=int, default=50, metavar="N",
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
        "--log-dir", "-ld", type=str, default="./logs", metavar="STR",
        help="root directory to store all training logs in"
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
        "--hours-per-fold", "-hpf", type=int, default=4,
        help="maximum time in hours to spend training the model in each fold"
    )
    parser.add_argument(
        "--cuda", "-c", action="store_true", default=False,
        help="if enabled, check for GPUs and use them"
    )

    return parser


def train_segan_cv(args):
    # check if we are using CUDA and set accelerator, devices, strategy
    if args.cuda:
        if torch.cuda.is_available():
            accelerator = "gpu"
            num_devices = torch.cuda.device_count()
            strategy = "ddp" if num_devices > 1 else None
            print(f"CUDA enabled and available, using {num_devices} GPUs with strategy: {strategy}")
        else:
            raise RuntimeError("CUDA enabled but not available.")
    else:
        print("CUDA not requested, training on CPU.")
        accelerator = "cpu"
        num_devices = 1
        strategy = None

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

        print("=" * 40)
        print(f"FOLD {f+1} / {args.folds}")
        print("=" * 40)

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
            'input_channels': args.input_channels,
            'output_classes': args.output_channels,
            'num_filters': args.model_channels,
            'channels_per_group': args.channels_per_group,
            'upsample_mode': args.upsample_mode,
            'is_3d': args.is_3d
        }
        segmentor, discriminators = get_segmentor_and_discriminators(**model_kwargs)
        segmentor.float()
        for d in discriminators:
            d.float()

        # create the task
        task = SeGANTask(
            segmentor,
            discriminators,
            L1Loss(),
            learning_rate=args.learning_rate
        )

        # create loggers
        logger_kwargs = {
            "save_dir": args.log_dir,
            "name": args.label,
            "version": f"{args.version}_f{f}"
        }
        csv_logger = CSVLogger(**logger_kwargs)

        # create callbacks
        early_stopping = EarlyStopping(
            monitor='val_dsc_0',
            mode='max',
            patience=args.early_stopping_patience
        )

        # create a Trainer and fit the model
        csv_logger.log_hyperparams(args)
        trainer = Trainer(
            accelerator=accelerator,
            devices=num_devices,
            strategy=strategy,
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
            train_segan_cv(args)
            # if successful, set the flag to end the while loop
            training_complete = True
        except RuntimeError as err:
            # if we get a runtime error, check if it involves being out of memory
            if "out of memory" in str(err):
                # if the error was because we're out of memory, then we want to reduce the batch size
                if args.batch_size == 1:
                    # if the batch size is 1, we can't reduce it anymore so give up and raise the error
                    print("CUDA OOM with batch size = 1, reduce model complexity or use less workers.")
                    raise err
                else:
                    # if the batch size is not 1, divide it by 2 (with integer division), empty the cache, and let
                    # the while loop go around again
                    args.batch_size = args.batch_size // 2
                    torch.cuda.empty_cache()

                    print("=" * 40)
                    print(f"Training script crashed due to OOM on GPU. Batch size set to {args.batch_size} and "
                          f"retrying...")
                    print("=" * 40)
            else:
                # if the error did not have to do with being out of memory, raise it
                raise err


if __name__ == "__main__":
    main()
