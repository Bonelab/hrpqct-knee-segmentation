from __future__ import annotations

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace

from bonelab.util.echo_arguments import echo_arguments
from bonelab.util.registration_util import check_inputs_exist, check_for_output_overwrite, message_s

import numpy as np
import SimpleITK as sitk
import os
import yaml

from skimage.morphology import binary_dilation, binary_erosion, binary_closing, ball
from skimage.measure import label as sklabel
from skimage.filters import gaussian, median


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
    dilated_mask = binary_dilation(mask, footprint=ball(thickness))
    return (dilated_mask & (~mask)).astype(int)


def erode_and_subtract(mask: np.ndarray, thickness: int) -> np.ndarray:
    if not(isinstance(mask, np.ndarray)) or (len(mask.shape) != 3):
        raise ValueError("`mask` must be a 3D numpy array")
    if not(isinstance(thickness, int)) or (thickness < 0):
        raise ValueError("`thickness` must be a positive integer")
    eroded_mask = binary_erosion(mask, footprint=ball(thickness))
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
    bone_mask = np.logical_or(trabecular_bone_mask, subchondral_bone_plate_mask)
    message_s(f"Step 3: B   <- filter(B | ni={num_iterations_remove_islands}, ng={num_iterations_fill_gaps}) | Iteratively filtering bone mask", silent)
    bone_mask = iterative_filter(bone_mask, num_iterations_remove_islands, num_iterations_fill_gaps)
    message_s(f"Step 4: MSc <- B  ∩ (¬ erode(B | ne={min_subchondral_bone_plate_thickness})) | Eroding and subtracting bone mask to get minimum subchondral bone plate mask", silent)
    minimum_subchondral_bone_plate_mask = erode_and_subtract(bone_mask, min_subchondral_bone_plate_thickness)
    message_s("Step 5: Tb  <- Tb ∩ (¬ MSc) | Subtracting the minimum subchondral bone plate mask from the trabecular mask", silent)
    trabecular_bone_mask = np.logical_and(trabecular_bone_mask, np.logical_not(minimum_subchondral_bone_plate_mask))
    message_s("Step 6: Sc  <- B  ∩ (¬ Tb) | Subtracting the trabecular mask from the bone mask to get the subchondral bone plate mask", silent)
    subchondral_bone_plate_mask = np.logical_and(bone_mask, np.logical_not(trabecular_bone_mask))
    message_s(f"Step 7: Sc  <- close(Sc, nc={subchondral_bone_plate_closing}) | Performing closing on subchondral bone plate mask", silent)
    subchondral_bone_plate_mask = binary_closing(subchondral_bone_plate_mask, ball(subchondral_bone_plate_closing))
    message_s("Step 8: Tb  <- Tb  ∩ (¬ Sc) | Subtracting the subchondral bone plate mask from the trabecular mask", silent)
    trabecular_bone_mask = np.logical_and(trabecular_bone_mask, np.logical_not(subchondral_bone_plate_mask))
    return subchondral_bone_plate_mask.astype(int), trabecular_bone_mask.astype(int)


def postprocess_segmentation(args: Namespace):
    print(echo_arguments("Post-process segmentation", vars(args)))
    # check inputs exist
    check_inputs_exist(
        [args.mask],
        args.silent
    )
    # generate filenames for outputs
    yaml_fn = os.path.join(args.output_dir, f"{args.output_label}_postprocessed_mask.yaml")
    post_model_mask_fn = os.path.join(args.output_dir, f"{args.output_label}_postprocessed_mask.nii.gz")
    # check for output overwrite
    check_for_output_overwrite(
        [yaml_fn, post_model_mask_fn],
        args.overwrite, args.silent
    )
    message_s("Writing yaml...", args.silent)
    with open(yaml_fn, "w") as f:
        yaml.dump(vars(args), f)
    message_s("Reading in mask...", args.silent)
    mask_sitk = sitk.ReadImage(args.mask)
    message_s("Converting mask to a numpy array...", args.silent)
    mask = sitk.GetArrayFromImage(mask_sitk)
    subchondral_bone_plate_mask = (mask == args.model_subchondral_bone_plate_class).astype(int)
    trabecular_bone_mask = (mask == args.model_trabecular_bone_class).astype(int)
    message_s("Post-processing mask...", args.silent)
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
    message_s("Writing post-processed mask...", args.silent)
    post_model_mask_sitk = sitk.GetImageFromArray(post_model_mask)
    post_model_mask_sitk.CopyInformation(mask_sitk)
    sitk.WriteImage(sitk.Cast(post_model_mask_sitk, sitk.sitkInt32), post_model_mask_fn)


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description='This script takes a mask and performs a series of '
                    'morphological operations to clean up the mask. The input mask should be a multi-class mask where '
                    'the optional parameters {model-subchondral-bone-plate-class} and {model-trabecular-bone-class} '
                    'specify the class labels for the subchondral bone plate and trabecular bone in the mask. '
                    'The output mask will have the following class labels: 0-background, 1-subchondral bone plate, '
                    '2-trabecular bone. The output mask will be saved to '
                    '{output_dir}/{output_label}_postprocessed_mask.nii.gz and a yaml file containing the arguments '
                    'supplied to this script will be saved to {output_dir}/{output_label}_postprocessed_mask.yaml.',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("mask", type=str, help="The input mask.")
    parser.add_argument("output_dir", type=str, help="The output directory.")
    parser.add_argument("output_label", type=str, help="The output label.")
    parser.add_argument(
        "--model-subchondral-bone-plate-class", "-msbp", type=int, default=0, metavar="N",
        help="the class label for the subchondral bone plate in the model mask"
    )
    parser.add_argument(
        "--model-trabecular-bone-class", "-mtb", type=int, default=1, metavar="N",
        help="the class label for the trabecular bone in the model mask"
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
    parser.add_argument("--overwrite", "-ow", action="store_true", help="Overwrite output files if they exist.")
    parser.add_argument("--silent", "-s", action="store_true", help="Silence all terminal output.")
    return parser


def main():
    args = create_parser().parse_args()
    postprocess_segmentation(args)


if __name__ == "__main__":
    main()
