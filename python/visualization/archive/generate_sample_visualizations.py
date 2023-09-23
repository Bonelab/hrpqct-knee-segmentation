from __future__ import annotations

from argparse import ArgumentParser
import torch.nn
from torch.utils.data import ConcatDataset
from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax
import yaml
from typing import Callable, List, Union
from monai.networks.nets.unet import UNet
from monai.networks.nets.unetr import UNETR
from monai.networks.nets.basic_unetplusplus import BasicUNetPlusPlus
from monai.networks.nets.segresnet import SegResNetVAE
from monai.transforms import VoteEnsemble, MeanEnsemble
from matplotlib import pyplot as plt
import random
import numpy as np

from bonelab.util.echo_arguments import echo_arguments
from blpytorchlightning.dataset_components.datasets.PickledDataset import PickledDataset
from blpytorchlightning.models.SeGAN import get_segmentor_and_discriminators
from blpytorchlightning.tasks.SeGANTask import SeGANTask
from blpytorchlightning.tasks.SegmentationTask import SegmentationTask
from blpytorchlightning.tasks.SegResNetVAETask import SegResNetVAETask


def calculate_dsc(x: np.ndarray, y: np.ndarray) -> float:
    return 2 * np.sum(x & y) / (np.sum(x) + np.sum(y))


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Generate sample visualizations using prepped data and a set of trained models. Rather than using"
                    "all command-line arguments, this script loads arguments from a yaml file. Check in "
                    "yamls/visualize_knee_models.yaml for an example."
    )
    parser.add_argument("yaml_fn", type=str, help="path to yaml file from which to read arguments")

    '''
    parser_cl = subparsers.add_parser("cl", help="parse args from command line")
    parser_cl.add_argument("--hparams", "-hp", type=str, nargs="+", required=True, help="path to hparams file for the trained model")
    parser_cl.add_argument("--checkpoints", "-c", type=str, nargs="+", required=True, help="path to the checkpoint of the trained model")
    parser_cl.add_argument("--labels", "-l", type=str, nargs="+", help="label, or labels, to put on plots")
    parser_cl.add_argument("--data_dirs", "-dd", type=str, nargs="+", required=True, help="list of directories to pull data from")
    parser_cl.add_argument("--num-plots", "-np", type=int, default=5, metavar="N", help="number of plots to generate")
    parser_cl.add_argument("--cort-color", "-cc", type=str, default="Greens", help="colormap to use for the cortical mask")
    parser_cl.add_argument("--trab-color", "-tc", type=str, default="Blues", help="colormap to use for the trabecular mask")
    parser_cl.add_argument("--error-color", "-ec", type=str, default="Reds", help="colormap to use for errors")
    parser_cl.add_argument(
        "--video", "-v", action="store_true", default=False,
        help="set this flag to make a video, otherwise the visualization will be of the central slice"
    )
    parser_cl.add_argument("--dim", "-d", type=int, default=0, help="the dimension to slice into for the image/video")
    parser_cl.add_argument(
        "--ensemble-mode", "-em", type=str, default=None,
        help="Can be set to `mean` to take the mean of model raw predictions, or `vote` to do a voting ensemble "
             "of model-predicted segmentations. If not set, each model will be used to make separate predictions."
    )
    '''

    return parser


# we need a factory function for creating a loss function that can be used for the unet++
def create_unetplusplus_loss_function(loss_function):
    def unetplusplus_loss_function(y_hat_list, y):
        loss = 0
        for y_hat in y_hat_list:
            loss += loss_function(y_hat, y)
        return loss
    return unetplusplus_loss_function


def load_unet_checkpoint(hparams: dict, checkpoint_fn: str) -> SegmentationTask:
    # create the model
    model_kwargs = {
        "spatial_dims": 3 if hparams["is_3d"] else 2,
        "in_channels": hparams["input_channels"],
        "out_channels": hparams["output_channels"],
    }
    if hparams["dropout"] < 0 or hparams["dropout"] > 1:
        raise ValueError("dropout must be between 0 and 1")
    if hparams["model_architecture"] == "unet":
        if len(hparams["model_channels"]) < 2:
            raise ValueError("model channels must be sequence of integers of at least length 2")
        model_kwagrs["channels"] = hparams["model_channels"]
        model_kwargs["strides"] = [1 for _ in range(len(hparams["model_channels"]) - 1)]
        model_kwargs["dropout"] = hparams["dropout"]
        model = UNet(**model_kwargs)
    elif hparams["model_architecture"] == "attention-unet":
        if len(args["model_channels"]) < 2:
            raise ValueError("model channels must be sequence of integers of at least length 2")
        model_kwargs["channels"] = hparams["model_channels"]
        model_kwargs["strides"] = [1 for _ in range(len(hparams["model_channels"]) - 1)]
        model_kwargs["dropout"] = hparams["dropout"]
        model = AttentionUnet(**model_kwargs)
    elif hparams["model_architecture"] == "unet-r":
        if hparams["image_size"] is None:
            raise ValueError("if model architecture set to `unet-r`, you must specify image size")
        if hparams["is_3d"] and len(args["image_size"]) != 3:
            raise ValueError("if 3D, image_size must be integer or length-3 sequence of integers")
        if not hparams["is_3d"] and len(args["image_size"]) != 2:
            raise ValueError("if not 3D, image_size must be integer or length-2 sequence of integers")
        model_kwargs["img_size"] = hparams["image_size"]
        model_kwargs["dropout_rate"] = hparams["dropout"]
        model_kwargs["feature_size"] = hparams["unet_r_feature_size"]
        model_kwargs["hidden_size"] = hparams["unet_r_hidden_size"]
        model_kwargs["mlp_dim"] = hparams["unet_r_mlp_dim"]
        model_kwargs["num_heads"] = hparams["unet_r_num_heads"]
        model = UNETR(**model_kwargs)
    elif hparams["model_architecture"] == "unet++":
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

    # create loss function
    loss_function = CrossEntropyLoss()
    if hparams["model_architecture"] == "unet++":
        loss_function = create_unetplusplus_loss_function(loss_function)

    # create the task
    return SegmentationTask.load_from_checkpoint(
        checkpoint_path=checkpoint_fn,
        model=model, loss_function=loss_function,
        learning_rate=hparams["learning_rate"],
    )


def load_segresnetvae_checkpoint(hparams: dict, checkpoint_fn: str) -> SegResNetVAETask:
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

    return SegResNetVAETask.load_from_checkpoint(
        checkpoint_path=checkpoint_fn,
        model=model, loss_function=loss_function,
        learning_rate=hparams["learning_rate"]
    )


def load_segan_checkpoint(hparams: dict, checkpoint_fn: str) -> SeGANTask:
    pass


def load_checkpoint(hparam_fn: str, checkpoint_fn: str) -> Union[SegmentationTask, SeGANTask, SegResNetVAETask]:
    with open(hparam_fn) as f:
        hparams = yaml.safe_load(f)

    # figure out what model architecture we are dealing with
    if "blocks_down" in hparams.keys():
        return load_segresnetvae_checkpoint(hparams, checkpoint_fn)
    elif "model_architecture" in hparams.keys():
        return load_unet_checkpoint(hparams, checkpoint_fn)
    else:
        return load_segan_checkpoint(hparams, checkpoint_fn)


def load_checkpoints(args: dict) -> List[Union[SegmentationTask, SeGANTask, SegResNetVAETask]]:
    tasks = []
    for hparam, checkpoint in zip(args["hparams"], args["checkpoints"]):
        tasks.append(load_checkpoint(hparam, checkpoint))
    return tasks


def create_ensemble_predictions(mode: str) -> Callable:
    if mode == "mean":
        ensembler = MeanEnsemble()

        def ensemble_predictions(
                tasks: List[Union[SegmentationTask, SeGANTask, SegResNetVAETask]],
                img: torch.Tensor
        ) -> torch.Tensor:
            # ensemble then argmax
            preds = []
            for task in tasks:
                with torch.no_grad():
                    pred = task(img.unsqueeze(0))
                if isinstance(pred, list):
                    pred = pred[0]
                preds.append(softmax(pred, dim=1))
            return torch.argmax(ensembler(preds), dim=1)

        return ensemble_predictions
    if mode == "vote":
        ensembler = VoteEnsemble()

        def ensemble_predictions(
                tasks: List[Union[SegmentationTask, SeGANTask, SegResNetVAETask]],
                img: torch.Tensor
        ) -> torch.Tensor:
            # argmax then ensemble
            preds = []
            for task in tasks:
                with torch.no_grad():
                    pred = task(img.unsqueeze(0))
                if isinstance(pred, list):
                    pred = pred[0]
                preds.append(torch.argmax(pred, dim=1))
            return ensembler(preds)

        return ensemble_predictions
    else:
        raise ValueError("`ensemble_mode` must be either `mean` or `vote`")


def generate_sample_visualizations(args: dict):
    print(echo_arguments("Generate Sample Visualizations", args))

    # error checking
    if not(len(args["hparams"]) == len(args["checkpoints"])):
        raise ValueError("did not get same number of hparam files as checkpoints, make sure they match up...")

    datasets = []
    for data_dir in args["data_dirs"]:
        datasets.append(PickledDataset(data_dir))
    dataset = ConcatDataset(datasets)

    tasks = load_checkpoints(args)

    ensemble_predictions = create_ensemble_predictions(mode=args["ensemble_mode"])

    fig, axs = plt.subplots(3, args["num_plots"])

    # reference masks on the top row, predictions on the middle row, errors on the bottom row
    ref_ax = axs[0]
    pred_ax = axs[1]
    error_ax = axs[2]

    for rax, pax, eax in zip(ref_ax, pred_ax, error_ax):
        # pull a random sample from the dataset for each column
        img, mask = dataset[random.randint(0, len(dataset) - 1)]
        mask_pred = ensemble_predictions(tasks, img)
        #with torch.no_grad():
        #    mask_pred = tasks[0](img.unsqueeze(0))
        #if isinstance(mask_pred, list):
        #    mask_pred = mask_pred[0]
        #mask_pred = torch.argmax(mask_pred, dim=1)

        img = img.numpy()[0, ...]
        cort = (mask == 0).numpy()
        trab = (mask == 1).numpy()

        cort_pred = (mask_pred == 0).numpy()[0, ...]
        trab_pred = (mask_pred == 1).numpy()[0, ...]

        cort_dsc = calculate_dsc(cort, cort_pred)
        trab_dsc = calculate_dsc(trab, trab_pred)

        errors = (cort_pred != cort) | (trab_pred != trab)

        s = [slice(None), slice(None), slice(None)]
        s[args["dim"]] = img.shape[args["dim"]]//2
        s = tuple(s)

        for ax in [rax, pax, eax]:
            ax.imshow(img[s], vmin=-1, vmax=1, cmap="gist_gray")
            ax.axis("off")

        rax.set_title(f"Sc.DSC: {cort_dsc:0.2f} \nTb.DSC: {trab_dsc:0.2f}")

        rax.imshow(cort[s], vmin=0, vmax=1, cmap=args["cort_color"], alpha=0.5 * cort[s].astype(float))
        rax.imshow(trab[s], vmin=0, vmax=1, cmap=args["trab_color"], alpha=0.5 * trab[s].astype(float))

        pax.imshow(cort_pred[s], vmin=0, vmax=1, cmap=args["cort_color"], alpha=0.5 * cort_pred[s].astype(float))
        pax.imshow(trab_pred[s], vmin=0, vmax=1, cmap=args["trab_color"], alpha=0.5 * trab_pred[s].astype(float))

        eax.imshow(errors[s], vmin=0, vmax=1, cmap=args["error_color"], alpha=0.5 * errors[s].astype(float))

    plt.tight_layout()
    plt.show()


def main():
    args = create_parser().parse_args()
    with open(args.yaml_fn) as f:
        args = yaml.safe_load(f)
    generate_sample_visualizations(args)


if __name__ == "__main__":
    main()
