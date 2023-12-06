from __future__ import annotations

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace

from bonelab.util.echo_arguments import echo_arguments
from bonelab.util.registration_util import check_inputs_exist, check_for_output_overwrite, message_s
from blpytorchlightning.tasks.SegmentationTask import SegmentationTask
from blpytorchlightning.tasks.SegResNetVAETask import SegResNetVAETask
from blpytorchlightning.tasks.SeGANTask import SeGANTask
from blpytorchlightning.models.SeGAN import get_segmentor_and_discriminators

import math
import numpy as np
import SimpleITK as sitk
import torch
import yaml
import os
from tqdm import tqdm, trange
from torch.nn import L1Loss, CrossEntropyLoss
from monai.networks.nets.unet import UNet
from monai.networks.nets.unetr import UNETR
from monai.networks.nets.basic_unetplusplus import BasicUNetPlusPlus
from monai.networks.nets.segresnet import SegResNetVAE
from monai.inferers import SlidingWindowInferer


class EnsembleSegmentationModel:
    def __init__(self,
                 models: List[Union[SegmentationTask, SeGANTask, SegResNetVAETask]],
                 inferer: SlidingWindowInferer,
                 silent: bool
                 ):
        self._models = models
        self._inferer = inferer
        self._silent = silent

    @property
    def models(self) -> List[Union[SegmentationTask, SeGANTask, SegResNetVAETask]]:
        return self._models

    @property
    def inferer(self) -> SlidingWindowInferer:
        return self._inferer

    @property
    def silent(self) -> bool:
        return self._silent

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
        y_hat = 0
        for i, model in enumerate(self.models):
            message_s(f"Performing inference with model {i+1} of {len(self.models)}...", self.silent)
            with torch.no_grad():
                pred = self.inferer(image, model)
            if isinstance(pred, list) or isinstance(pred, tuple):
                pred = sum(pred)
            y_hat += pred.squeeze(0).squeeze(0)
        return torch.argmax(y_hat.squeeze(0), dim=0).cpu().numpy()


def create_unetplusplus_loss_function(loss_function):
    def unetplusplus_loss_function(y_hat_list: List[torch.Tensor], y: torch.Tensor) -> torch.Tensor:
        loss = 0
        for y_hat in y_hat_list:
            loss += loss_function(y_hat, y)
        return loss
    return unetplusplus_loss_function


def create_unet(hparams: dict) -> torch.nn.Module:
    model_kwargs = {
        "spatial_dims": 3 if hparams["is_3d"] else 2,
        "in_channels": hparams["input_channels"],
        "out_channels": hparams["output_channels"],
    }
    if hparams["dropout"] < 0 or hparams["dropout"] > 1:
        raise ValueError("dropout must be between 0 and 1")
    if hparams.get("model_architecture") == "unet" or hparams.get("model_architecture") is None:
        if len(hparams["model_channels"]) < 2:
            raise ValueError("model channels must be sequence of integers of at least length 2")
        model_kwargs["channels"] = hparams["model_channels"]
        model_kwargs["strides"] = [1 for _ in range(len(hparams["model_channels"]) - 1)]
        model_kwargs["dropout"] = hparams["dropout"]
        model = UNet(**model_kwargs)
    elif hparams.get("model_architecture") == "attention-unet":
        if len(hparams["model_channels"]) < 2:
            raise ValueError("model channels must be sequence of integers of at least length 2")
        model_kwargs["channels"] = hparams["model_channels"]
        model_kwargs["strides"] = [1 for _ in range(len(hparams["model_channels"]) - 1)]
        model_kwargs["dropout"] = hparams["dropout"]
        model = AttentionUnet(**model_kwargs)
    elif hparams.get("model_architecture") == "unet-r":
        if hparams["image_size"] is None:
            raise ValueError("if model architecture set to `unet-r`, you must specify image size")
        if hparams["is_3d"] and len(hparams["image_size"]) != 3:
            raise ValueError("if 3D, image_size must be integer or length-3 sequence of integers")
        if not hparams["is_3d"] and len(hparams["image_size"]) != 2:
            raise ValueError("if not 3D, image_size must be integer or length-2 sequence of integers")
        model_kwargs["img_size"] = hparams["image_size"]
        model_kwargs["dropout_rate"] = hparams["dropout"]
        model_kwargs["feature_size"] = hparams["unet_r_feature_size"]
        model_kwargs["hidden_size"] = hparams["unet_r_hidden_size"]
        model_kwargs["mlp_dim"] = hparams["unet_r_mlp_dim"]
        model_kwargs["num_heads"] = hparams["unet_r_num_heads"]
        model = UNETR(**model_kwargs)
    elif hparams.get("model_architecture") == "unet++":
        if len(hparams["model_channels"]) != 6:
            raise ValueError("if model architecture set to `unet++`, model channels must be length-6 sequence of "
                             "integers")
        model_kwargs["features"] = hparams["model_channels"]
        model_kwargs["dropout"] = hparams["dropout"]
        model = BasicUNetPlusPlus(**model_kwargs)
    else:
        raise ValueError(f"model architecture must be `unet`, `attention-unet`, `unet++`, or `unet-r`, "
                         f"given {hparams['model_architecture']}")
    model.float()
    return model


def load_task(
        hparams_fn: str,
        checkpoint_fn: str,
        model_type: str,
        device: torch.device
) -> Union[SegmentationTask, SeGANTask, SegResNetVAETask]:

    with open(hparams_fn) as f:
        hparams = yaml.safe_load(f)

    if model_type == "unet":
        model = create_unet(hparams)
        loss_function = CrossEntropyLoss()
        if hparams.get("model_architecture") == "unet++":
            loss_function = create_unetplusplus_loss_function(loss_function)
        task = SegmentationTask(
            model=model, loss_function=loss_function,
            learning_rate=hparams["learning_rate"]
        )
        task = task.load_from_checkpoint(
            checkpoint_fn,
            model=model, loss_function=loss_function,
            learning_rate=hparams["learning_rate"]
        )
        task.to(device)
        return task

    elif model_type == "segan":
        model_kwargs = {
            "input_channels": hparams["input_channels"],
            "output_classes": hparams["output_channels"],
            "num_filters": hparams["model_channels"],
            "channels_per_group": hparams["channels_per_group"],
            "upsample_mode": hparams["upsample_mode"],
            "is_3d": hparams["is_3d"]
        }
        segmentor, discriminators = get_segmentor_and_discriminators(**model_kwargs)
        segmentor.float()
        for d in discriminators:
            d.float()
        task = SeGANTask(segmentor, discriminators, L1Loss(), learning_rate=hparams["learning_rate"])
        task.load_from_checkpoint(
            checkpoint_fn,
            segmentor, discriminators, L1Loss(), learning_rate=hparams["learning_rate"]
        )
        task.to(device)
        return task

    elif model_type == "segresnetvae":
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
        task = SegResNetVAETask(model, CrossEntropyLoss(), learning_rate=hparams["learning_rate"])
        task.load_from_checkpoint(
            checkpoint_fn,
            model=model,
            loss_function=CrossEntropyLoss(),
            learning_rate=hparams["learning_rate"]
        )
        task.to(device)
        return task

    else:
        raise ValueError(f"model type must be `unet`, `segan`, or `segresnetvae`, given {model_type}")


def inference_ensemble(args: Namespace):
    print(echo_arguments("Ensemble model inference", vars(args)))
    message_s("Checking if cuda was requested and available...", args.silent)
    if args.cuda:
        if torch.cuda.is_available():
            message_s("cuda requested and available, using cuda...", args.silent)
            device = torch.device("cuda")
        else:
            message_s("cuda requested but unavailable, using cpu...", args.silent)
            device = torch.device("cpu")
    else:
        message_s("cuda not requested, using cpu...", args.silent)
        device = torch.device("cpu")
    check_inputs_exist(
        [args.image] + args.hparams_filenames + args.checkpoint_filenames,
        args.silent
    )
    yaml_fn = os.path.join(args.output_dir, f"{args.output_label}_ensemble_inference.yaml")
    model_mask_fn = os.path.join(args.output_dir, f"{args.output_label}_ensemble_inference_mask.nii.gz")
    check_for_output_overwrite(
        [yaml_fn, model_mask_fn],
        args.overwrite, args.silent
    )
    message_s("Writing yaml...", args.silent)
    with open(yaml_fn, "w") as f:
        yaml.dump(vars(args), f)
    message_s("Reading in image...", args.silent)
    image_sitk = sitk.ReadImage(args.image)
    message_s("Converting image to a numpy array...", args.silent)
    image = sitk.GetArrayFromImage(image_sitk)
    message_s("Rescaling image from densities to [-1, +1] range...", args.silent)
    image = np.minimum(np.maximum(image, args.min_density), args.max_density)
    image = (2 * image - args.max_density - args.min_density) / (args.max_density - args.min_density)
    message_s("Constructing ensemble model...", args.silent)
    ensemble_model = EnsembleSegmentationModel(
        [
            load_task(hparams_fn, checkpoint_fn, model_type, device)
            for hparams_fn, checkpoint_fn, model_type in zip(
                args.hparams_filenames, args.checkpoint_filenames, args.model_types,
            )
        ],
        SlidingWindowInferer(
            roi_size=args.patch_width,
            sw_batch_size=args.batch_size,
            overlap=args.overlap,
            mode="gaussian",
            sw_device=device,
            device="cpu",
            progress=(~args.silent)
        ),
        args.silent
    )
    message_s("Performing inference on image...", args.silent)
    model_mask = ensemble_model(image)
    message_s("Writing model mask...", args.silent)
    model_mask_sitk = sitk.GetImageFromArray(model_mask)
    model_mask_sitk.CopyInformation(image_sitk)
    sitk.WriteImage(sitk.Cast(model_mask_sitk, sitk.sitkInt32), model_mask_fn)


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="This script takes in an image and a set of models and performs "
                    "inference on the image with each model, then combines the results into a single mask. "
                    "Inference is performed using a sliding window method with configurable overlap and patch width. "
                    "NOTE: The models should have been trained with the same data pre-processing as will be applied "
                    "in this script - specifically, linear rescaling using {min_density} and {max_density} as the "
                    "input range. "
                    "The models can be of different types, but must all be able to accept the same input image size "
                    "and have the same output mask size. The out put mask will be saved to "
                    "{output_dir}/{output_label}_ensemble_inference.nii.gz, along with a yaml file, "
                    "{output_dir}/{output_label}_ensemble_inference.yaml, that contains all arguments supplied to "
                    "this script. The mask will contain the raw output from the model."
                    "It is highly recommended to run this on a server or workstation with a GPU and to use the --cuda "
                    "flag. Knee images are large and inference using multiple models with sliding-window inference "
                    "can be extremely slow on a CPU.",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("image", type=str, help="The image nii file.")
    parser.add_argument("output_dir", type=str, help="The output directory.")
    parser.add_argument("output_label", type=str, help="The output label.")
    parser.add_argument(
        "--hparams-filenames", "-hf", type=str, nargs="+", required=True, metavar="FN",
        help="The filenames of the hparams files for the models to use for inference."
    )
    parser.add_argument(
        "--checkpoint-filenames", "-cf", type=str, nargs="+", required=True, metavar="FN",
        help="The filenames of the checkpoint files for the models to use for inference."
    )
    parser.add_argument(
        "--model-types", "-mt", choices=["unet", "segan", "segresnetvae"], nargs="+", required=True, metavar="MT",
        help="The types of models to use for inference."
    )
    parser.add_argument(
        "--patch-width", "-pw", type=int, default=64, metavar="N",
        help="width of cubic patches to split image into for inference (NOTE: for some models, you have to match this"
             "to how you trained the model, e.g. UNETR)"
    )
    parser.add_argument(
        "--overlap", "-o", type=float, default=0.25, metavar="D",
        help="overlap between patches when performing inference"
    )
    parser.add_argument(
        '--min-density', '-mind', type=float, default=-400, metavar='D',
        help='minimum physiologically relevant density in the image [mg HA/ccm]. ensure this is the same value you '
             'used when training the models.'
    )
    parser.add_argument(
        '--max-density', '-maxd', type=float, default=1400, metavar='D',
        help='maximum physiologically relevant density in the image [mg HA/ccm]. ensure this is the same value you '
             'used when training the models.'
    )
    parser.add_argument(
        "--batch-size", "-bs", type=int, default=32, metavar="BS",
        help="batch size to use for inference"
    )
    parser.add_argument("--cuda", "-c", action="store_true", help="Use CUDA if available.")
    parser.add_argument("--overwrite", "-ow", action="store_true", help="Overwrite output files if they exist.")
    parser.add_argument("--silent", "-s", action="store_true", help="Silence all terminal output.")
    return parser


def main():
    args = create_parser().parse_args()
    inference_ensemble(args)


if __name__ == "__main__":
    main()
