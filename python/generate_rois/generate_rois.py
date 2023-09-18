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
from skimage.morphology import binary_dilation, binary_erosion, binary_closing, ball
from skimage.measure import label as sklabel
from skimage.filters import gaussian, median


class EnsembleSegmentationModel:
    def __init__(self,
                 models: List[Union[SegmentationTask, SeGANTask, SegResNetVAETask]],
                 device: torch.device,
                 inferer: SlidingWindowInferer,
                 silent: bool
                 ):
        self._models = models
        self._device = device
        self._inferer = inferer
        self._silent = silent

    @property
    def models(self) -> List[Union[SegmentationTask, SeGANTask, SegResNetVAETask]]:
        return self._models

    @property
    def device(self) -> torch.device:
        return self._device

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
                pred = pred[-1]
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


def keep_largest_connected_component_skimage(mask: np.ndarray, background: bool = False) -> np.ndarray:
    if not(isinstance(mask, np.ndarray)) or (len(mask.shape) != 3):
        raise ValueError("`mask` must be a 3D numpy array")
    mask = ~mask if background else mask
    labelled_mask = sklabel(mask, background=0)
    component_counts = np.bincount(labelled_mask.flat)
    if len(component_counts) < 2:
        return mask
    mask = labelled_mask == np.argmax(component_counts[1:]) + 1
    mask = ~mask if background else mask
    return mask.astype(int)


def remove_islands_from_mask(mask: np.ndarray, erosion_dilation: int = 1) -> np.ndarray:
    if not(isinstance(erosion_dilation, int)) or (erosion_dilation < 0):
        raise ValueError("`erosion_dilation` must be a positive integer")
    if not(isinstance(mask, np.ndarray)) or (len(mask.shape) != 3):
        raise ValueError("`mask` must be a 3D numpy array")
    mask = np.pad(mask, ((1, 1), (1, 1), (1, 1)), mode='constant')
    binary_erosion(mask, footprint=ball(erosion_dilation), out=mask)
    mask = keep_largest_connected_component_skimage(mask.astype(int), background=False)
    binary_dilation(mask, footprint=ball(erosion_dilation), out=mask)
    return mask[1:-1, 1:-1, 1:-1].astype(int)


def fill_in_gaps_in_mask(mask: np.ndarray, dilation_erosion: int = 1) -> np.ndarray:
    if not(isinstance(dilation_erosion, int)) or (dilation_erosion < 0):
        raise ValueError("`dilation_erosion` must be a positive integer")
    if not(isinstance(mask, np.ndarray)) or (len(mask.shape) != 3):
        raise ValueError("`mask` must be a 3D numpy array")
    pad_width = 2 * dilation_erosion if (dilation_erosion > 0) else None
    if pad_width:
        pad_width = 2 * dilation_erosion
        mask = np.pad(mask, ((pad_width, pad_width), (pad_width, pad_width), (pad_width, pad_width)), mode='constant')
    binary_dilation(mask, footprint=ball(dilation_erosion), out=mask)
    mask = keep_largest_connected_component_skimage(mask.astype(int), background=True)
    binary_erosion(mask, footprint=ball(dilation_erosion), out=mask)
    if pad_width:
        mask = mask[pad_width:-pad_width, pad_width:-pad_width, pad_width:-pad_width]
    return keep_largest_connected_component_skimage(mask.astype(int), background=True)


def iterative_filter(mask: np.ndarray, n_islands: int, n_gaps: int) -> np.ndarray:
    if not(isinstance(mask, np.ndarray)) or (len(mask.shape) != 3):
        raise ValueError("`mask` must be a 3D numpy array")
    if not(isinstance(n_islands, int)) or (n_islands < 0):
        raise ValueError("`n_islands` must be a positive integer")
    if not(isinstance(n_gaps, int)) or (n_gaps < 0):
        raise ValueError("`n_gaps` must be a positive integer")
    for n in range(1, min(n_islands, n_gaps) + 1):
        mask = remove_islands_from_mask(mask, erosion_dilation=n)
        mask = fill_in_gaps_in_mask(mask, dilation_erosion=n)
    if n_islands > n_gaps:
        mask = remove_islands_from_mask(mask, erosion_dilation=n_islands)
    elif n_gaps > n_islands:
        mask = fill_in_gaps_in_mask(mask, dilation_erosion=n_gaps)
    return mask


def dilate_and_subtract(mask: np.ndarray, thickness: int) -> np.ndarray:
    if not(isinstance(mask, np.ndarray)) or (len(mask.shape) != 3):
        raise ValueError("`mask` must be a 3D numpy array")
    if not(isinstance(thickness, int)) or (thickness < 0):
        raise ValueError("`thickness` must be a positive integer")
    dilated_mask = mask.copy()
    binary_dilation(dilated_mask, footprint=ball(thickness), out=dilated_mask)
    return (dilated_mask & (~mask)).astype(int)


def erode_and_subtract(mask: np.ndarray, thickness: int) -> np.ndarray:
    if not(isinstance(mask, np.ndarray)) or (len(mask.shape) != 3):
        raise ValueError("`mask` must be a 3D numpy array")
    if not(isinstance(thickness, int)) or (thickness < 0):
        raise ValueError("`thickness` must be a positive integer")
    eroded_mask = mask.copy()
    binary_erosion(eroded_mask, footprint=ball(thickness), out=eroded_mask)
    return ((~eroded_mask) & mask).astype(int)


def extract_bone(image, threshold: float = -0.25):
    if not(isinstance(mask, np.ndarray)) or (len(mask.shape) != 3):
        raise ValueError("`mask` must be a 3D numpy array")
    bone_mask = image >= threshold
    bone_mask = median(bone_mask, selem=np.ones((3, 3, 1)))
    bone_mask = remove_islands_from_mask(bone_mask, erosion_dilation=3)
    bone_mask = fill_in_gaps_in_mask(bone_mask, dilation_erosion=15)
    return bone_mask


def postprocess_model_masks(
        subchondral_bone_plate_mask: np.ndarray,
        trabecular_bone_mask: np.ndarray,
        min_subchondral_bone_plate_thickness: int = 4,
        num_iterations_remove_islands: int = 2,
        num_iterations_fill_gaps: int = 2,
        subchondral_bone_plate_closing: int = 4,
        silent: bool = False
) -> np.ndarray:
    message_s("", silent)
    message_s(f"Step 1: Tb  <- filter(Tb | ni={num_iterations_remove_islands}, ng={num_iterations_fill_gaps}) | Iteratively filtering trabecular mask", silent)
    trabecular_bone_mask = iterative_filter(trabecular_bone_mask, num_iterations_remove_islands, num_iterations_fill_gaps)
    message_s("Step 2: B   <- Tb ∪ Sc | Combining filtered trabecular and subchondral bone plate masks into bone mask", silent)
    bone_mask = trabecular_bone_mask | subchondral_bone_plate_mask
    message_s(f"Step 3: B   <- filter(B | ni={num_iterations_remove_islands}, ng={num_iterations_fill_gaps}) | Iteratively filtering bone mask", silent)
    bone_mask = iterative_filter(bone_mask, num_iterations_remove_islands, num_iterations_fill_gaps)
    message_s(f"Step 4: MSc <- B  ∩ (¬ erode(B | ne={min_subchondral_bone_plate_thickness})) | Eroding and subtracting bone mask to get minimum subchondral bone plate mask", silent)
    minimum_subchondral_bone_plate_mask = erode_and_subtract(bone_mask, min_subchondral_bone_plate_thickness)
    message_s("Step 5: Tb  <- Tb ∩ (¬ MSc) | Subtracting the minimum subchondral bone plate mask from the trabecular mask", silent)
    trabecular_bone_mask = trabecular_bone_mask & (~minimum_subchondral_bone_plate_mask)
    message_s("Step 6: Sc  <- B  ∩ (¬ Tb) | Subtracting the trabecular mask from the bone mask to get the subchondral bone plate mask", silent)
    subchondral_bone_plate_mask = bone_mask & (~trabecular_bone_mask)
    message_s(f"Step 7: Sc  <- close(Sc, nc={subchondral_bone_plate_closing}) | Performing closing on subchondral bone plate mask", silent)
    subchondral_bone_plate_mask = binary_closing(subchondral_bone_plate_mask, ball(subchondral_bone_plate_closing))
    message_s("Step 8: Tb  <- Tb  ∩ (¬ Sc) | Subtracting the subchondral bone plate mask from the trabecular mask", silent)
    trabecular_bone_mask = trabecular_bone_mask & (~subchondral_bone_plate_mask)
    return subchondral_bone_plate_mask.astype(int), trabecular_bone_mask.astype(int)


def get_regional_subchondral_bone_plate_mask(
        subchondral_bone_plate_mask: np.ndarray,
        roi_mask: np.ndarray,
        bone: str
) -> np.ndarray:
    if not isinstance(subchondral_bone_plate_mask, np.ndarray):
        raise ValueError("`subchondral_bone_plate_mask` must be a numpy array")
    if not isinstance(roi_mask, np.ndarray):
        raise ValueError("`roi_mask` must be a numpy array")
    if not isinstance(bone, str):
        raise ValueError("`bone` must be a string")
    if bone == "femur":
        keep_lower = True
    elif bone == "tibia":
        keep_lower = False
    else:
        raise ValueError(f"bone must be `femur` or `tibia`, given {bone}")
    mask = subchondral_bone_plate_mask & roi_mask
    labelled_mask = sklabel(mask, background=0)
    if np.max(labelled_mask) == 0:
        raise ValueError("No subchondral bone plate detected in the contact region ROI")
    z_centers = [np.mean(np.nonzero(labelled_mask == label)[0]) for label in range(1, np.max(labelled_mask) + 1)]
    return (labelled_mask == (np.argmin(z_centers) + 1 if keep_lower else np.argmax(z_centers) + 1)).astype(int)


def generate_periarticular_rois_from_bone_plate_and_trabecular_masks(
        subchondral_bone_plate_mask: np.ndarray,
        trabecular_bone_mask: np.ndarray,
        dilation_kernel_up: np.ndarray,
        dilation_kernel_down: np.ndarray,
        compartment_depth: int,
        silent: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # extract a sub-image to work with so it's faster
    message_s("Extracting sub-image to work with...", silent)

    # combine the bone plate and trabecular bone masks to get the bone mask, then find the top layer of bone
    message_s("Finding top layer of bone...", silent)
    top_layer_mask = (
            binary_dilation(subchondral_bone_plate_mask, dilation_kernel_up)
            & ~subchondral_bone_plate_mask
    ).astype(int)
    # dilate down into the bone to get the shallow mask
    message_s("Dilating down into the bone to get the shallow mask...", silent)
    shallow_mask = top_layer_mask
    for _ in trange(compartment_depth, disable=silent):
        shallow_mask = binary_dilation(shallow_mask, dilation_kernel_down)
    shallow_mask = (
            shallow_mask & trabecular_bone_mask
            & ~subchondral_bone_plate_mask & ~top_layer_mask
    ).astype(int)
    # dilate down into the bone to get the mid mask
    message_s("Dilating down into the bone to get the mid mask...", silent)
    mid_mask = top_layer_mask
    for _ in trange(2 * compartment_depth, disable=silent):
        mid_mask = binary_dilation(mid_mask, dilation_kernel_down)
    mid_mask = (
            mid_mask & trabecular_bone_mask
            & ~shallow_mask & ~subchondral_bone_plate_mask & ~top_layer_mask
    ).astype(int)
    # dilate down into the bone to get the deep mask
    message_s("Dilating down into the bone to get the deep mask...", silent)
    deep_mask = top_layer_mask
    for _ in trange(3 * compartment_depth, disable=silent):
        deep_mask = binary_dilation(deep_mask, dilation_kernel_down)
    deep_mask = (
            deep_mask & trabecular_bone_mask
            & ~mid_mask & ~shallow_mask & ~subchondral_bone_plate_mask & ~top_layer_mask
    ).astype(int)
    return subchondral_bone_plate_mask, shallow_mask, mid_mask, deep_mask


def generate_rois(args: Namespace):
    print(echo_arguments("ROI Generation", vars(args)))
    # check if we asked for cuda and if available
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
    # check inputs exist
    check_inputs_exist(
        [args.image, args.atlas_mask] + args.hparams_filenames + args.checkpoint_filenames,
        args.silent
    )
    # generate filenames for outputs
    yaml_fn = os.path.join(args.output_dir, f"{args.output_label}.yaml")
    model_mask_fn = os.path.join(args.output_dir, f"{args.output_label}_model_mask.nii.gz")
    post_model_mask_fn = os.path.join(args.output_dir, f"{args.output_label}_post_model_mask.nii.gz")
    allrois_mask_fn = os.path.join(args.output_dir, f"{args.output_label}_allrois_mask.nii.gz")
    if args.bone == "femur":
        medial_site_codes = args.femur_medial_site_codes
        lateral_site_codes = args.femur_lateral_site_codes
    elif args.bone == "tibia":
        medial_site_codes = args.tibia_medial_site_codes
        lateral_site_codes = args.tibia_lateral_site_codes
    else:
        raise ValueError(f"bone must be `femur` or `tibia`, given {args.bone}")
    medial_roi_mask_fns = [
        os.path.join(args.output_dir, f"{args.output_label}_roi{code}_mask.nii.gz")
        for code in medial_site_codes
    ]
    lateral_roi_mask_fns = [
        os.path.join(args.output_dir, f"{args.output_label}_roi{code}_mask.nii.gz")
        for code in lateral_site_codes
    ]
    # check for output overwrite
    check_for_output_overwrite(
        [yaml_fn, model_mask_fn, allrois_mask_fn] + medial_roi_mask_fns + lateral_roi_mask_fns,
        args.overwrite, args.silent
    )
    # write yaml
    message_s("Writing yaml...", args.silent)
    with open(yaml_fn, "w") as f:
        yaml.dump(vars(args), f)
    # read in the image and the atlas mask
    message_s("Reading in image and atlas mask...", args.silent)
    image_sitk = sitk.ReadImage(args.image)
    atlas_mask_sitk = sitk.ReadImage(args.atlas_mask)
    # dilate the atlas mask in the axial direction to ensure that it contains the subchondral bone plate, for both
    # the lateral and medial sides
    message_s("Dilating atlas mask...", args.silent)
    atlas_mask_sitk = sitk.BinaryDilate(
        sitk.BinaryDilate(
            atlas_mask_sitk,
            [0, 0, args.axial_dilation_footprint],
            foregroundValue=args.lateral_atlas_code
        ),
        [0, 0, args.axial_dilation_footprint],
        foregroundValue=args.medial_atlas_code,
    )
    # convert the image and atlas mask to numpy arrays
    message_s("Converting image and atlas mask to numpy arrays...", args.silent)
    image = sitk.GetArrayFromImage(image_sitk)
    atlas_mask = sitk.GetArrayFromImage(atlas_mask_sitk)
    # rescale the image from densities to the [-1, +1] range
    message_s("Rescaling image from densities to [-1, +1] range...", args.silent)
    image = np.minimum(np.maximum(image, args.min_density), args.max_density)
    image = (2 * image - args.max_density - args.min_density) / (args.max_density - args.min_density)
    # construct the ensemble model
    message_s("Constructing ensemble model...", args.silent)
    ensemble_model = EnsembleSegmentationModel(
        [
            load_task(hparams_fn, checkpoint_fn, model_type, device)
            for hparams_fn, checkpoint_fn, model_type in zip(
                args.hparams_filenames, args.checkpoint_filenames, args.model_types,
            )
        ],
        device,
        SlidingWindowInferer(
            roi_size=args.patch_width,
            sw_batch_size=1,
            overlap=args.overlap,
            mode="gaussian",
            sw_device=device,
            device=device,
            progress=(~args.silent)
        ),
        args.silent
    )
    # create dilation kernels for constructing the periarticular ROIs
    message_s("Creating dilation kernels...", args.silent)
    dilation_kernel_up = np.zeros((3, 3, 3), dtype=int)
    dilation_kernel_up[1, 1, 1] = 1
    dilation_kernel_down = np.zeros((3, 3, 3), dtype=int)
    dilation_kernel_down[1, 1, 1] = 1
    if args.bone == "femur":
        dilation_kernel_up[0, 1, 1] = 1
        dilation_kernel_down[2, 1, 1] = 1
    elif args.bone == "tibia":
        dilation_kernel_up[2, 1, 1] = 1
        dilation_kernel_down[0, 1, 1] = 1
    else:
        raise ValueError(f"bone must be `femur` or `tibia`, given {args.bone}")
    message_s("Performing inference on image...", args.silent)
    model_mask = ensemble_model(image)
    message_s("Postprocessing model masks...", args.silent)
    subchondral_bone_plate_mask = (model_mask == args.model_subchondral_bone_plate_class).astype(int)
    trabecular_bone_mask = (model_mask == args.model_trabecular_bone_class).astype(int)
    model_mask = subchondral_bone_plate_mask + 2 * trabecular_bone_mask
    post_subchondral_bone_plate_mask, post_trabecular_bone_mask = postprocess_model_masks(
        subchondral_bone_plate_mask,
        trabecular_bone_mask,
        args.minimum_subchondral_bone_plate_thickness,
        args.num_iterations_remove_islands,
        args.num_iterations_fill_gaps,
        args.subchondral_bone_plate_closing,
        args.silent
    )
    post_model_mask = post_subchondral_bone_plate_mask + 2 * post_trabecular_bone_mask
    medial_subchondral_bone_plate_mask = get_regional_subchondral_bone_plate_mask(
        post_subchondral_bone_plate_mask,
        atlas_mask == args.medial_atlas_code,
        args.bone
    )
    medial_trabecular_bone_mask = (
        trabecular_bone_mask
        & (atlas_mask == args.medial_atlas_code)
    ).astype(int)
    message_s("Generating medial ROIs...", args.silent)
    medial_roi_masks = generate_periarticular_rois_from_bone_plate_and_trabecular_masks(
        medial_subchondral_bone_plate_mask,
        medial_trabecular_bone_mask,
        dilation_kernel_up,
        dilation_kernel_down,
        args.compartment_depth,
        args.silent
    )
    message_s("Writing model mask...", args.silent)
    model_mask_sitk = sitk.GetImageFromArray(model_mask)
    model_mask_sitk.CopyInformation(atlas_mask_sitk)
    sitk.WriteImage(sitk.Cast(model_mask_sitk, sitk.sitkInt32), model_mask_fn)
    message_s("Writing post-processed model mask...", args.silent)
    post_model_mask_sitk = sitk.GetImageFromArray(post_model_mask)
    post_model_mask_sitk.CopyInformation(atlas_mask_sitk)
    sitk.WriteImage(sitk.Cast(post_model_mask_sitk, sitk.sitkInt32), post_model_mask_fn)
    # create an all rois mask to start adding to
    all_rois_mask = sitk.Image(*model_mask_sitk.GetSize(), model_mask_sitk.GetPixelID())
    all_rois_mask = sitk.Cast(all_rois_mask, sitk.sitkInt32)
    all_rois_mask.CopyInformation(model_mask_sitk)
    message_s("Writing medial ROI masks...", args.silent)
    for mask, fn, msc in zip(medial_roi_masks, medial_roi_mask_fns, medial_site_codes):
        mask_sitk = sitk.GetImageFromArray(mask)
        mask_sitk.CopyInformation(atlas_mask_sitk)
        sitk.WriteImage(sitk.Cast(mask_sitk, sitk.sitkInt32), fn)
        all_rois_mask += msc * sitk.Cast(mask_sitk, sitk.sitkInt32)

    message_s("Writing all rois mask...", args.silent)
    sitk.WriteImage(all_rois_mask, allrois_mask_fn)


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description='ROI Generation Script',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("image", type=str, help="The image nii file.")
    parser.add_argument("bone", choices=["femur", "tibia"], help="The bone to generate ROIs for.")
    parser.add_argument("atlas_mask", type=str, help="The atlas mask nii file.")
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
        "--axial-dilation-footprint", "-adf", default=20, type=int, metavar="N",
        help="the footprint to use for the axial dilation of the atlas mask"
    )
    parser.add_argument(
        "--femur-lateral-site-codes", "-flsc", default=[17, 13, 14, 15], type=int, nargs=4, metavar="N",
        help="the site codes for the lateral ROIs in the femur, the order should be: "
             "bone plate, shallow, mid, deep"
    )
    parser.add_argument(
        "--femur-medial-site-codes", "-fmsc", default=[16, 10, 11, 12], type=int, nargs=4, metavar="N",
        help="the site codes for the medial ROIs in the femur, the order should be: "
             "bone plate, shallow, mid, deep"
    )
    parser.add_argument(
        "--tibia-lateral-site-codes", "-tlsc", default=[37, 33, 34, 35], type=int, nargs=4, metavar="N",
        help="the site codes for the lateral ROIs in the tibia, the order should be: "
             "bone plate, shallow, mid, deep"
    )
    parser.add_argument(
        "--tibia-medial-site-codes", "-tmsc", default=[36, 30, 31, 32], type=int, nargs=4, metavar="N",
        help="the site codes for the medial ROIs in the tibia, the order should be: "
             "bone plate, shallow, mid, deep"
    )
    parser.add_argument(
        "--lateral-atlas-code", "-lac", default=2, type=int, metavar="N",
        help="site code used in the atlas mask for the combined lateral VOI"
    )
    parser.add_argument(
        "--medial-atlas-code", "-mac", default=1, type=int, metavar="N",
        help="site code used in the atlas mask for the combined medial VOI"
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
        "--model-subchondral-bone-plate-class", "-msbp", type=int, default=0, metavar="N",
        help="the class label for the subchondral bone plate in the model mask"
    )
    parser.add_argument(
        "--model-trabecular-bone-class", "-mtb", type=int, default=1, metavar="N",
        help="the class label for the trabecular bone in the model mask"
    )
    parser.add_argument(
        '--min-density', '-mind', type=float, default=-400, metavar='D',
        help='minimum physiologically relevant density in the image [mg HA/ccm]'
    )
    parser.add_argument(
        '--max-density', '-maxd', type=float, default=1400, metavar='D',
        help='maximum physiologically relevant density in the image [mg HA/ccm]'
    )
    parser.add_argument(
        "--compartment-depth", "-cd", type=int, default=41, metavar="N",
        help="depth of shallow, mid, deep compartments, in voxels"
    )
    parser.add_argument(
        "--minimum-subchondral-bone-plate-thickness", "-msbpt", type=int, default=4, metavar="N",
        help="minimum thickness of the subchondral bone plate, in voxels"
    )
    parser.add_argument(
        "--num-iterations-remove-islands", "-niri", type=int, default=2, metavar="N",
        help="number of iterations of island removal to perform"
    )
    parser.add_argument(
        "--num-iterations-fill-gaps", "-nifg", type=int, default=2, metavar="N",
        help="number of iterations of gap filling to perform"
    )
    parser.add_argument(
        "--subchondral-bone-plate-closing", "-sbpc", type=int, default=4, metavar="N",
        help="radius of structural element when performing closing on the subchondral bone plate mask"
    )
    parser.add_argument("--cuda", "-c", action="store_true", help="Use CUDA if available.")
    parser.add_argument("--overwrite", "-ow", action="store_true", help="Overwrite output files if they exist.")
    parser.add_argument("--silent", "-s", action="store_true", help="Silence all terminal output.")
    return parser


def main():
    args = create_parser().parse_args()
    generate_rois(args)


if __name__ == "__main__":
    main()
