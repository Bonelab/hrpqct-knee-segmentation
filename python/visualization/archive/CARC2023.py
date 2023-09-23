""" Code to generate visualizations for CARC2023 Research Presentation Days  presentation """

import numpy as np
from blpytorchlightning.dataset_components.datasets.PickledDataset import PickledDataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from blpytorchlightning.tasks.SegmentationTask import SegmentationTask
from monai.networks.nets.unet import UNet
from monai.networks.nets.attentionunet import AttentionUnet
from monai.networks.nets.unetr import UNETR
from monai.networks.nets.basic_unetplusplus import BasicUNetPlusPlus
from torch.nn import CrossEntropyLoss
from glob import glob

import torch
import os
import yaml


# we need a factory function for creating a loss function that can be used for the unet++
def create_unetplusplus_loss_function(loss_function):
    def unetplusplus_loss_function(y_hat_list, y):
        loss = 0
        for y_hat in y_hat_list:
            loss += loss_function(y_hat, y)
        return loss
    return unetplusplus_loss_function


# model creation got a bit crazy, separate into function for readability
def create_model(ref_hparams) -> torch.nn.Module():
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
    else:
        raise ValueError(f"model architecture must be `unet`, `attention-unet`, `unet++`, or `unet-r`, "
                         f"given {ref_hparams['model_architecture']}")

    model.float()

    return model


def get_task(log_dir, label, version):

    with open(os.path.join(log_dir, label, "ref_hparams.yaml")) as f:
        ref_hparams = yaml.safe_load(f)

    model = create_model(ref_hparams)

    # create loss function
    loss_function = CrossEntropyLoss()
    if ref_hparams.get("model_architecture") == "unet++":
        loss_function = create_unetplusplus_loss_function(loss_function)

    checkpoint_path = glob(
        os.path.join(
            log_dir,
            label,
            version,
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

    return task


def make_video(filename, samples, figsize, panning_dimension, opacity, num_frames, fps, interval):

    fig, axs = plt.subplots(2, len(samples), figsize=figsize)

    def animate(i: int) -> None:
        slicing_list = [slice(None), slice(None), slice(None)]
        slicing_list[panning_dimension] = i

        for i, (k, v) in enumerate(samples.items()):
            image = samples[k]["x"][tuple(slicing_list)]
            true_cortical_mask = (samples[k]["y"][tuple(slicing_list)] == 0).astype(float)
            pred_cortical_mask = (samples[k]["y_hat"][tuple(slicing_list)] == 0).astype(float)

            error = (true_cortical_mask != pred_cortical_mask).astype(float)

            axs[0][i].set_title(k)
            axs[0][i].clear()
            axs[0][i].imshow(image, cmap="gist_gray", vmin=-1, vmax=1)
            axs[0][i].imshow(true_cortical_mask, cmap="Greens", alpha=opacity * true_cortical_mask)
            axs[0][i].set_frame_on(False)
            axs[0][i].axes.get_xaxis().set_visible(False)
            axs[0][i].axes.get_yaxis().set_visible(False)

            axs[1][i].clear()
            axs[1][i].imshow(image, cmap="gist_gray", vmin=-1, vmax=1)
            axs[1][i].imshow(pred_cortical_mask, cmap="Blues", alpha=opacity * pred_cortical_mask)
            axs[1][i].imshow(error, cmap="Reds", alpha=opacity * error)
            axs[1][i].set_frame_on(False)
            axs[1][i].axes.get_xaxis().set_visible(False)
            axs[1][i].axes.get_yaxis().set_visible(False)

        fig.tight_layout()

    animation_frames = np.arange(num_frames)

    anim = FuncAnimation(fig, animate, frames=animation_frames, interval=interval)
    plt.show()
    response = input("Enter anything if you're happy and want to save, if you don't type anything it will not save.")
    if response == "":
        return
    try:
        ffmpeg_writer = FFMpegWriter(fps=fps)
        anim.save(f"{filename}.mp4", writer=ffmpeg_writer)
    except FileNotFoundError:
        print("`ffmpeg` not found, saving as a gif. Run `conda install ffmpeg` if you want an mp4.")
        anim.save(f"{filename}.gif", writer="imagemagick")


def main():

    # parameters for loading the model and task
    log_dir = "/Users/nathanneeteson/Projects/hrpqct-knee-segmentation/from_arc"
    label = "unetr_3d_knee_cv"
    version = "17172956_f0"

    # parameters for loading the data to visualize
    data_dirs = {
        "Femur, Medial": "/Users/nathanneeteson/Documents/Data/Images/SALTAC/visit_1/femur/medial_3d_patches",
        "Femur, Lateral": "/Users/nathanneeteson/Documents/Data/Images/SALTAC/visit_1/femur/lateral_3d_patches",
        "Tibia, Medial": "/Users/nathanneeteson/Documents/Data/Images/SALTAC/visit_1/tibia/medial_3d_patches",
        "Tibia, Lateral": "/Users/nathanneeteson/Documents/Data/Images/SALTAC/visit_1/tibia/lateral_3d_patches",
    }

    # plotting parameters
    st = (slice(None), 32, slice(None))
    opacity = 0.5
    figsize = (8, 4)

    # video parameters
    filename = "/Users/nathanneeteson/Documents/Conferences/2023/CARC2023/presentation/sample_v2"
    panning_dimension = 1
    num_frames = 64
    fps = 30
    interval = 100

    task = get_task(log_dir, label, version)

    dataloader_kwargs = {
        'batch_size': 1,
        'num_workers': 1,
        'pin_memory': True,
        'persistent_workers': True
    }

    samples = {}
    for k, v in data_dirs.items():
        samples[k] = {}
        x, y = next(iter(DataLoader(PickledDataset(v), **dataloader_kwargs)))
        y_hat = torch.argmax(task(x), dim=1)
        samples[k]["x"], samples[k]["y"] = x[0, 0, :, :, :].numpy(), y[0, :, :, :].numpy()
        samples[k]["y_hat"] = y_hat[0, :, :, :].numpy()

    fig, axs = plt.subplots(2, len(samples), figsize=figsize)

    for i, (k, v) in enumerate(samples.items()):

        true_cortical_mask = (samples[k]["y"][st] == 0).astype(float)
        pred_cortical_mask = (samples[k]["y_hat"][st] == 0).astype(float)

        error = (true_cortical_mask != pred_cortical_mask).astype(float)

        axs[0][i].set_title(k)

        axs[0][i].imshow(samples[k]["x"][st], cmap="gist_gray", vmin=-1, vmax=1)
        axs[0][i].imshow(true_cortical_mask, cmap="Greens", alpha=opacity * true_cortical_mask)

        axs[1][i].imshow(samples[k]["x"][st], cmap="gist_gray", vmin=-1, vmax=1)
        axs[1][i].imshow(pred_cortical_mask, cmap="Blues", alpha=opacity * pred_cortical_mask)
        axs[1][i].imshow(error, cmap="Reds", alpha=opacity * error)

    for ax in axs:
        for a in ax:
            a.set_axis_off()
    plt.tight_layout()

    plt.show()

    make_video(filename, samples, figsize, panning_dimension, opacity, num_frames, fps, interval)






if __name__ == "__main__":
    main()
