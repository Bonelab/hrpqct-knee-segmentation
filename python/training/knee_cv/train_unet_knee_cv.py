from argparse import Namespace
import numpy as np
import os
import torch
import yaml
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
from parser import create_parser


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
    # load the hyperparameters from file
    # with open(os.path.join(args.log_dir, args.reference_label, args.reference_version, "hparams.yaml")) as f:
    #     hparams = yaml.safe_load(f)
    with open(os.path.join(args.log_dir, args.reference_label, args.reference_version, "ref_hparams.yaml")) as f:
        ref_hparams = yaml.safe_load(f)

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
        print(f"FOLD {f + 1} / {args.folds}")
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
            "spatial_dims": 3 if ref_hparams["is_3d"] else 2,
            "in_channels": ref_hparams["input_channels"],
            "out_channels": ref_hparams["output_channels"],
        }
        if ref_hparams["dropout"] < 0 or ref_hparams["dropout"] > 1:
            raise ValueError("dropout must be between 0 and 1")
        if ref_hparams.get("model_architecture") == "unet" or ref_hparams.get("model_architecture") is None:
            if len(ref_hparams["model_channels"]) < 2:
                raise ValueError("model channels must be sequence of integers of at least length 2")
            model_kwargs["channels"] = ref_hparams["model_channels"]
            model_kwargs["strides"] = [1 for _ in range(len(ref_hparams["model_channels"]) - 1)]
            model_kwargs["dropout"] = ref_hparams["dropout"]
            model = UNet(**model_kwargs)
        elif ref_hparams.get("model_architecture") == "attention-unet":
            if len(ref_hparams["model_channels"]) < 2:
                raise ValueError("model channels must be sequence of integers of at least length 2")
            model_kwargs["channels"] = ref_hparams["model_channels"]
            model_kwargs["strides"] = [1 for _ in range(len(ref_hparams["model_channels"]) - 1)]
            model_kwargs["dropout"] = ref_hparams["dropout"]
            model = AttentionUnet(**model_kwargs)
        elif ref_hparams.get("model_architecture") == "unet-r":
            if ref_hparams["image_size"] is None:
                raise ValueError("if model architecture set to `unet-r`, you must specify image size")
            if ref_hparams["is_3d"] and len(ref_hparams["image_size"]) != 3:
                raise ValueError("if 3D, image_size must be integer or length-3 sequence of integers")
            if not ref_hparams["is_3d"] and len(ref_hparams["image_size"]) != 2:
                raise ValueError("if not 3D, image_size must be integer or length-2 sequence of integers")
            model_kwargs["img_size"] = ref_hparams["image_size"]
            model_kwargs["dropout_rate"] = ref_hparams["dropout"]
            model_kwargs["feature_size"] = ref_hparams["unet_r_feature_size"]
            model_kwargs["hidden_size"] = ref_hparams["unet_r_hidden_size"]
            model_kwargs["mlp_dim"] = ref_hparams["unet_r_mlp_dim"]
            model_kwargs["num_heads"] = ref_hparams["unet_r_num_heads"]
            model = UNETR(**model_kwargs)
        elif ref_hparams.get("model_architecture") == "unet++":
            if len(ref_hparams["model_channels"]) != 6:
                raise ValueError("if model architecture set to `unet++`, model channels must be length-6 sequence of "
                                 "integers")
            model_kwargs["features"] = ref_hparams["model_channels"]
            model_kwargs["dropout"] = ref_hparams["dropout"]
            model = BasicUNetPlusPlus(**model_kwargs)
            if strategy == "ddp_find_unused_parameters_false":
                strategy = "ddp"
                print(f"using `unet++`, so changing strategy to {strategy}")
        else:
            raise ValueError(f"model architecture must be `unet`, `attention-unet`, `unet++`, or `unet-r`, "
                             f"given {ref_hparams['model_architecture']}")

        model.float()

        # create loss function
        loss_function = CrossEntropyLoss()
        if ref_hparams.get("model_architecture") == "unet++":
            loss_function = create_unetplusplus_loss_function(loss_function)

        checkpoint_path = glob(
            os.path.join(
                args.log_dir,
                args.reference_label,
                args.reference_version,
                "checkpoints",
                "*.ckpt"
            )
        )[0]
        print(f"Loading model and task from: {checkpoint_path}")

        # create the task
        task = SegmentationTask(
            model=model, loss_function=loss_function,
            learning_rate=ref_hparams["learning_rate"]
        )

        task = task.load_from_checkpoint(
            checkpoint_path,
            model=model, loss_function=loss_function,
            learning_rate=ref_hparams["learning_rate"]
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



def main():
    # get parameters from command line
    args = create_parser("UNet").parse_args()

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
