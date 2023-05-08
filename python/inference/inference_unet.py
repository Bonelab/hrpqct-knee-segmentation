""" Code to perform inference on a new image using a trained model """
from __future__ import annotations
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
import numpy as np
import vtk
import vtkbone
import pytorch_lightning as pl
from bonelab.util.echo_arguments import echo_arguments
from bonelab.util.vtk_util import vtkImageData_to_numpy, numpy_to_vtkImageData
from bonelab.util.aim_calibration_header import get_aim_density_equation
from bonelab.io.vtk_helpers import handle_filetype_writing_special_cases
from blpytorchlightning.tasks.SegmentationTask import SegmentationTask
from monai.networks.nets.unet import UNet
from monai.networks.nets.attentionunet import AttentionUnet
from monai.networks.nets.unetr import UNETR
from monai.networks.nets.basic_unetplusplus import BasicUNetPlusPlus
from torch.nn import CrossEntropyLoss
from glob import glob
from tqdm import tqdm
from datetime import datetime

import torch
import os
import yaml

#TODO: Add option for patch overlap so padding doesn't influence segmentation accuracy at edges of patches

def create_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description='Segmentation Inference Script',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("hparams_filename", type=str, help="filename of the hparams file for the trained model")
    parser.add_argument("checkpoint_filename", type=str, help="the filename of the checkpoint file to load")
    parser.add_argument("image_filename", type=str, help="filename of image to do inference on")
    parser.add_argument("scbp_filename", type=str, help="filename to save inferred subchondral bone plate "
                                                        "segmentation to")
    parser.add_argument("trab_filename", type=str, help="filename to save inferred trabecular bone "
                                                        "segmentation to")
    parser.add_argument(
        "--patch-width", "-pw", type=int, default=64, metavar="N",
        help="width of cubic patches to split image into for inference (NOTE: for some models, you have to match this"
             "to how you trained the model, e.g. UNETR)"
    )
    parser.add_argument(
        '--min-density', '-mind', type=float, default=-400, metavar='D',
        help='minimum physiologically relevant density in the image [mg HA/ccm]'
    )
    parser.add_argument(
        '--max-density', '-maxd', type=float, default=1400, metavar='D',
        help='maximum physiologically relevant density in the image [mg HA/ccm]'
    )
    return parser


# we need a factory function for creating a loss function that can be used for the unet++
def create_unetplusplus_loss_function(loss_function):
    def unetplusplus_loss_function(y_hat_list: List[torch.Tensor], y: torch.Tensor) -> torch.Tensor:
        loss = 0
        for y_hat in y_hat_list:
            loss += loss_function(y_hat, y)
        return loss
    return unetplusplus_loss_function


# model creation got a bit crazy, separate into function for readability
def create_model(hparams: dict) -> torch.nn.Module():
    # create the model
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


def get_task(hparams_fn, checkpoint_fn):

    with open(hparams_fn) as f:
        hparams = yaml.safe_load(f)

    model = create_model(hparams)

    # create loss function
    loss_function = CrossEntropyLoss()
    if hparams.get("model_architecture") == "unet++":
        loss_function = create_unetplusplus_loss_function(loss_function)

    # create the task
    task = SegmentationTask(
        model=model, loss_function=loss_function,
        learning_rate=hparams["learning_rate"]
    )

    task = task.load_from_checkpoint(
        checkpoint_fn,
        model=model, loss_function=loss_function,
        learning_rate=hparams["learning_rate"]
    )

    return task


def write_mask(fn: str, mask: np.ndarray, reader: vtkbone.vtkboneAIMReader, label: str):
    print(f"Updating processing log for {label} mask.")
    processing_log = (
        reader.GetProcessingLog()
        + os.linesep
        + f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initial {label} mask created."
    )
    print(f"Writing mask to {fn}")
    writer = vtkbone.vtkboneAIMWriter()
    writer.SetFileName(fn)
    writer.SetInputData(numpy_to_vtkImageData(
        127 * mask,
        spacing=reader.GetOutput().GetSpacing(),
        origin=reader.GetOutput().GetOrigin(),
        array_type=vtk.VTK_CHAR
    ))
    handle_filetype_writing_special_cases(
        writer,
        processing_log=processing_log
    )
    writer.Update()


def infer_segmentation(
        task: pl.LightningModule,
        img_fn: str,
        scbp_fn: str,
        trab_fn: str,
        patch_width: int,
        min_density: float,
        max_density: float
):
    # step 1: read image
    print(f"Reading image from {img_fn}")
    reader = vtkbone.vtkboneAIMReader()
    reader.DataOnCellsOff()
    reader.SetFileName(img_fn)
    reader.Update()

    # step 2: convert image to numpy array
    print("Converting to numpy.")
    image = vtkImageData_to_numpy(reader.GetOutput())

    # step 3: convert image to densities
    m, b = get_aim_density_equation(reader.GetProcessingLog())
    image = (m * image + b).astype(float)

    # step 4: rescale from densities to normalized range the model expects
    image = np.minimum(np.maximum(image, min_density), max_density)
    image = (2 * image - max_density - min_density) / (
            max_density - min_density
    )

    # step 5: pad image so all side lengths are a multiple of patch width
    print(f"Image has shape: {image.shape}, padding.")
    pad = [-s % patch_width for s in image.shape]
    image = np.pad(image, tuple([(p, 0) for p in pad]), mode="constant")
    mask = np.zeros_like(image)
    print(f"After padding, image has shape: {image.shape}")

    # step 6: perform inference one patch at a time
    ni, nj, nk = [s//patch_width for s in image.shape]
    print(f"Total number of patches: {ni*nj*nk}")
    for i, j, k in tqdm(np.ndindex(ni, nj, nk)):
        st = (
            slice(i * patch_width, (i + 1) * patch_width),
            slice(j * patch_width, (j + 1) * patch_width),
            slice(k * patch_width, (k + 1) * patch_width)
        )
        y_hat = task(torch.from_numpy(image[st]).unsqueeze(0).unsqueeze(0).float())
        if isinstance(y_hat, list):
            y_hat = y_hat[-1]
        mask[st] = torch.argmax(y_hat.squeeze(0), dim=0).numpy()
    print("Inference complete.")

    # step 7: trim back to original size
    print("Trimming mask to original shape...")
    mask = mask[pad[0]:, pad[1]:, pad[2]:]
    print(f"Trimmed mask has shape: {mask.shape}")

    # step 8: write masks
    write_mask(scbp_fn, mask == 0, reader, "subchondral bone plate")
    write_mask(trab_fn, mask == 1, reader, "trabecular bone")


def main() -> None:
    # get parameters from command line
    args = create_parser().parse_args()
    print()
    print(echo_arguments("Inference - UNet", vars(args)))
    task = get_task(args.hparams_filename, args.checkpoint_filename)
    print("Loaded model from checkpoint successfully. Starting inference.")
    infer_segmentation(
        task,
        args.image_filename, args.scbp_filename, args.trab_filename,
        args.patch_width, args.min_density, args.max_density
    )
    
    
if __name__ == "__main__":
    main()
