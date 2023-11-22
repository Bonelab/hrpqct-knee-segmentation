import numpy as np
import os
import torch
import yaml
from torch.nn import CrossEntropyLoss
from torchmetrics import Dice
from torch.utils.data import DataLoader, ConcatDataset, Subset
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from blpytorchlightning.tasks.SegResNetVAETask import SegResNetVAETask
from blpytorchlightning.dataset_components.datasets.PickledDataset import PickledDataset
from blpytorchlightning.loss_functions.DiceLoss import DiceLoss
from monai.networks.nets.segresnet import SegResNetVAE
from parser import create_parser


def train_segresnetvae_cv(args):
    # load the hyperparameters from file
    with open(os.path.join(args.log_dir, args.reference_label, args.reference_version, args.hparams_fn)) as f:
        hparams = yaml.safe_load(f)

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

    # dataloader standard kwargs
    dataloader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'pin_memory': True,
        'persistent_workers': True,
        'shuffle': True
    }

    # create dataloader
    dataloader = DataLoader(dataset, **dataloader_kwargs)

    # create model
    model_kwargs = {
        "spatial_dims": 3 if hparams["is_3d"] else 2,
        "input_image_size": hparams["image_size"],
        "in_channels": hparams["input_channels"],
        "out_channels": hparams["output_channels"],
        "dropout_prob": hparams["dropout"],
        "init_filters": hparams["init_filters"],
        "blocks_down": tuple(hparams["blocks_down"]),
        "blocks_up": tuple(hparams["blocks_up"]),
        "upsample_mode": "pixelshuffle"
    }
    model = SegResNetVAE(**model_kwargs)

    # create loss function
    loss_function = CrossEntropyLoss()

    # create task
    task = SegResNetVAETask(
        model, loss_function,
        learning_rate=hparams["learning_rate"]
    )

    # create loggers
    logger_kwargs = {
        "save_dir": args.log_dir,
        "name": args.label,
        "version": args.version
    }
    csv_logger = CSVLogger(**logger_kwargs)

    # create callbacks
    early_stopping = EarlyStopping(
        monitor='train_dsc_0',
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
        log_every_n_steps=args.log_step_interval,
        logger=csv_logger,
        callbacks=[early_stopping]
    )
    trainer.fit(task, dataloader)


def main() -> None:
    # get parameters from command line
    args = create_parser("SegResNetVAE").parse_args()

    training_complete = False

    # start the training attempt loop
    while not training_complete:
        try:
            # attempt to train at current batch size
            train_segresnetvae_cv(args)
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
