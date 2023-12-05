from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, ConcatDataset, Subset
from pytorch_lightning import Trainer
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from blpytorchlightning.tasks.SegmentationTask import SegmentationTask
from blpytorchlightning.dataset_components.datasets.PickledDataset import PickledDataset
from blpytorchlightning.dataset_components.transformers.TensorOneHotEncoder import TensorOneHotEncoder
from blpytorchlightning.loss_functions.DiceLoss import DiceLoss
from monai.networks.nets.unet import UNet
from monai.networks.nets.attentionunet import AttentionUnet
from monai.networks.nets.unetr import UNETR
from monai.networks.nets.basic_unetplusplus import BasicUNetPlusPlus
from glob import glob
from shutil import rmtree


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
        "--model-architecture", "-ma", type=str, default="unet", metavar="STR",
        help="model architecture to use, must be one of `unet`, `attention-unet`, `unet++` or `unet-r`"
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
        "--unet-r-feature-size", "-urfs", type=int, default="16", metavar="N",
        help="feature size for `unet-r`"
    )
    parser.add_argument(
        "--unet-r-hidden-size", "-urhs", type=int, default=768, metavar="N",
        help="hidden size for `unet-r`"
    )
    parser.add_argument(
        "--unet-r-mlp-dim", "-urmlp", type=int, default=3072, metavar="N",
        help="dimension of feed-forward layer for `unet-r`"
    )
    parser.add_argument(
        "--unet-r-num-heads", "-urnh", type=int, default=12, metavar="N",
        help="number of attention heads for `unet-r`"
    )
    parser.add_argument(
        '--is-3d', '-3d', action="store_true", default=False,
        help="set this flag to train a UNet with 3D convolutions for segmenting 3D image data"
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
    parser.add_argument(
        "--image-size", "-is", type=int, nargs="+", default=None,
        help="size of image, must specify if using `unet-r`"
    )
    parser.add_argument(
        "--dice-loss", "-dl", action="store_true", default=False,
        help="enable this flag to use Dice loss instead of Cross-Entropy"
    )

    return parser


# we need a factory function for creating a loss function that can be used for the unet++
def create_unetplusplus_loss_function(loss_function):
    def unetplusplus_loss_function(y_hat_list, y):
        loss = 0
        for y_hat in y_hat_list:
            loss += loss_function(y_hat, y)
        return loss
    return unetplusplus_loss_function


def train_unet_cv(args: Namespace) -> None:
    torch.set_float32_matmul_precision('medium')
    # check if we are using CUDA and set accelerator, devices, strategy
    if args.cuda:
        if torch.cuda.is_available():
            accelerator = "gpu"
            num_devices = torch.cuda.device_count()
            strategy = "ddp_find_unused_parameters_false" if num_devices > 1 else None
            print(f"CUDA enabled and available, using {num_devices} GPUs with strategy: {strategy}")
        else:
            raise RuntimeError("CUDA enabled but not available.")
    else:
        print("CUDA not requested, training on CPU.")
        accelerator = "cpu"
        num_devices = 1
        strategy = None

    # create datasets
    transformer = TensorOneHotEncoder(num_classes=args.output_channels) if args.dice_loss else None
    datasets = []
    for data_dir in args.data_dirs:
        datasets.append(PickledDataset(data_dir, transformer=transformer))
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
            "spatial_dims": 3 if args.is_3d else 2,
            "in_channels": args.input_channels,
            "out_channels": args.output_channels,
        }
        if args.dropout < 0 or args.dropout > 1:
            raise ValueError("dropout must be between 0 and 1")
        if args.model_architecture == "unet":
            if len(args.model_channels) < 2:
                raise ValueError("model channels must be sequence of integers of at least length 2")
            model_kwargs["channels"] = args.model_channels
            model_kwargs["strides"] = [1 for _ in range(len(args.model_channels) - 1)]
            model_kwargs["dropout"] = args.dropout
            model = UNet(**model_kwargs)
        elif args.model_architecture == "attention-unet":
            if len(args.model_channels) < 2:
                raise ValueError("model channels must be sequence of integers of at least length 2")
            model_kwargs["channels"] = args.model_channels
            model_kwargs["strides"] = [1 for _ in range(len(args.model_channels) - 1)]
            model_kwargs["dropout"] = args.dropout
            model = AttentionUnet(**model_kwargs)
        elif args.model_architecture == "unet-r":
            if args.image_size is None:
                raise ValueError("if model architecture set to `unet-r`, you must specify image size")
            if args.is_3d and len(args.image_size) != 3:
                raise ValueError("if 3D, image_size must be integer or length-3 sequence of integers")
            if not args.is_3d and len(args.image_size) != 2:
                raise ValueError("if not 3D, image_size must be integer or length-2 sequence of integers")
            model_kwargs["img_size"] = args.image_size
            model_kwargs["dropout_rate"] = args.dropout
            model_kwargs["feature_size"] = args.unet_r_feature_size
            model_kwargs["hidden_size"] = args.unet_r_hidden_size
            model_kwargs["mlp_dim"] = args.unet_r_mlp_dim
            model_kwargs["num_heads"] = args.unet_r_num_heads
            model = UNETR(**model_kwargs)
        elif args.model_architecture == "unet++":
            if len(args.model_channels) != 6:
                raise ValueError("if model architecture set to `unet++`, model channels must be length-6 sequence of "
                                 "integers")
            model_kwargs["features"] = args.model_channels
            model_kwargs["dropout"] = args.dropout
            model = BasicUNetPlusPlus(**model_kwargs)
            if strategy == "ddp_find_unused_parameters_false":
                strategy = "ddp"
                print(f"using `unet++`, so changing strategy to {strategy}")
        else:
            raise ValueError(f"model architecture must be `unet`, `attention-unet`, `unet++`, or `unet-r`, "
                             f"given {args.model_architecture}")

        model.float()

        # create loss function
        loss_function = DiceLoss() if args.dice_loss else CrossEntropyLoss()
        if args.model_architecture == "unet++":
            loss_function = create_unetplusplus_loss_function(loss_function)

        # create the task
        task = SegmentationTask(
            model, loss_function,
            learning_rate=args.learning_rate,
            ohe_targets=args.dice_loss
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
            train_unet_cv(args)
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
                    print("=" * 40)
                    print(f"Training script crashed due to OOM on GPU. Batch size set to {args.batch_size} and "
                          f"retrying...")
                    print("=" * 40)
            else:
                # if the error did not have to do with being out of memory, raise it
                raise err


if __name__ == "__main__":
    main()
