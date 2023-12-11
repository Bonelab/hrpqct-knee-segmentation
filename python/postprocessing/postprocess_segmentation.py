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


def expand_array_to_3d(array: np.ndarray, dim: int) -> np.ndarray:
    if dim == 0:
        return array[:, None, None]
    elif dim == 1:
        return array[None, :, None]
    elif dim == 2:
        return array[None, None, :]
    else:
        raise ValueError("`dim` must be 0, 1, or 2")


def create_efficient_3d_binary_operation(func: callable) -> callable:
    def efficient_3d_binary_operation(mask: np.ndarray, radius: int) -> np.ndarray:
        for dim in [0, 1, 2]:
            mask = func(mask, expand_array_to_3d(np.ones((2 * radius + 1,)), dim))
        return mask
    return efficient_3d_binary_operation


def efficient_3d_dilation(mask: np.ndarray, radius: int) -> np.ndarray:
    return create_efficient_3d_binary_operation(binary_dilation)(mask, radius)


def efficient_3d_erosion(mask: np.ndarray, radius: int) -> np.ndarray:
    return create_efficient_3d_binary_operation(binary_erosion)(mask, radius)


def efficient_3d_closing(mask: np.ndarray, radius: int) -> np.ndarray:
    return efficient_3d_dilation(efficient_3d_erosion(mask, radius), radius)


def efficient_3d_opening(mask: np.ndarray, radius: int) -> np.ndarray:
    return efficient_3d_erosion(efficient_3d_dilation(mask, radius), radius)


def keep_largest_connected_component_skimage(mask: np.ndarray, background: bool = False) -> np.ndarray:
    if not(isinstance(mask, np.ndarray)):
        raise ValueError("`mask` must be a 3D numpy array")
    mask = (1 - mask) if background else mask
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
    if erosion_dilation > 0:
        mask = efficient_3d_erosion(mask, erosion_dilation)
    mask = keep_largest_connected_component_skimage(mask.astype(int), background=False)
    if erosion_dilation > 0:
        mask = efficient_3d_dilation(mask, erosion_dilation)
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
    if dilation_erosion > 0:
        mask = efficient_3d_dilation(mask, dilation_erosion)
    mask = keep_largest_connected_component_skimage(mask.astype(int), background=True)
    if dilation_erosion > 0:
        mask = efficient_3d_erosion(mask, dilation_erosion)
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
    dilated_mask = efficient_3d_dilation(mask, thickness)
    return (dilated_mask & (~mask)).astype(int)


def erode_and_subtract(mask: np.ndarray, thickness: int) -> np.ndarray:
    if not(isinstance(mask, np.ndarray)) or (len(mask.shape) != 3):
        raise ValueError("`mask` must be a 3D numpy array")
    if not(isinstance(thickness, int)) or (thickness < 0):
        raise ValueError("`thickness` must be a positive integer")
    eroded_mask = efficient_3d_erosion(mask, thickness)
    return ((~eroded_mask) & mask).astype(int)


def postprocess_model_masks(
        subchondral_bone_plate_mask: np.ndarray,
        trabecular_bone_mask: np.ndarray,
        min_subchondral_bone_plate_thickness: int = 4,
        bone_fill_gaps_radius: int = 5,
        bone_remove_islands_radius: int = 4,
        trab_fill_gaps_radius: int = 5,
        silent: bool = False
) -> np.ndarray:
    message_s("", silent)
    message_s(f"B <- Tb ∪ Sc", silent)
    bone_mask = trabecular_bone_mask | subchondral_bone_plate_mask
    message_s(f"B <- fill_gaps(B | r={bone_fill_gaps_radius})", silent)
    bone_mask = fill_in_gaps_in_mask(bone_mask, dilation_erosion=bone_fill_gaps_radius)
    message_s(f"B <- remove_islands(B | r={bone_remove_islands_radius})", silent)
    bone_mask = remove_islands_from_mask(bone_mask, erosion_dilation=bone_remove_islands_radius)
    message_s(f"Sc <- Sc ∪ (B ∩ (¬ erode(B | r={min_subchondral_bone_plate_thickness})))", silent)
    subchondral_bone_plate_mask = (
            subchondral_bone_plate_mask
            | erode_and_subtract(bone_mask, min_subchondral_bone_plate_thickness)
    )
    message_s(f"Sc <- remove_islands(Sc | r=0)", silent)
    subchondral_bone_plate_mask = remove_islands_from_mask(subchondral_bone_plate_mask, 0)
    message_s(f"Tb <- B ∩ (¬ Sc)", silent)
    trabecular_bone_mask = bone_mask & (~subchondral_bone_plate_mask)
    message_s(
        f"Tb <- Tb ∪ (erode(fill_gaps(dilate(Tb | r={trab_fill_gaps_radius}) | r=0) | r={2*trab_fill_gaps_radius}))",
        silent
    )
    trabecular_bone_mask = (
        trabecular_bone_mask
        | efficient_3d_erosion(
            fill_in_gaps_in_mask(
                efficient_3d_dilation(
                    trabecular_bone_mask,
                    trab_fill_gaps_radius
                ),
                dilation_erosion=0
            ),
            2 * trab_fill_gaps_radius
        )
    )
    message_s(f"Sc <- Sc ∩ (¬ Tb)", silent)
    subchondral_bone_plate_mask = subchondral_bone_plate_mask & (~trabecular_bone_mask)
    return subchondral_bone_plate_mask.astype(int), trabecular_bone_mask.astype(int)


def keep_smaller_components(mask: np.ndarray) -> np.ndarray:
    return np.logical_and(mask, np.logical_not(keep_largest_connected_component_skimage(mask, background=False)))


def slice_wise_keep_smaller_components(mask: np.ndarray, dims: List[int], pad_amount: int = 5) -> np.ndarray:
    mask = np.pad(
        mask,
        ((pad_amount, pad_amount), (pad_amount, pad_amount), (pad_amount, pad_amount)),
        mode='constant',
        constant_values=1
    )
    out = np.zeros_like(mask)
    for dim in dims:
        for i in range(mask.shape[dim]):
            st = tuple([slice(None) if j != dim else i for j in range(len(mask.shape))])
            out[st] = np.logical_or(out[st], keep_smaller_components(mask[st]))
    return out[pad_amount:-pad_amount, pad_amount:-pad_amount, pad_amount:-pad_amount]


def segment_tunnel(
        cortical_mask: np.ndarray,
        trabecular_mask: np.ndarray,
        tunnel_min_size: int = 0,
        silent: bool = False
) -> np.ndarray:
    message_s("", silent)
    message_s(f"Step 1: B <- ¬(Sc ∪ Tb)", silent)
    background_mask = np.logical_not(np.logical_or(cortical_mask, trabecular_mask))
    message_s(f"Step 2: T <- slice_wise_keep_smaller_components(B)", silent)
    tunnel_mask = slice_wise_keep_smaller_components(background_mask, dims=[0, 1, 2])  # hard code to use all dimensions
    message_s(f"Step 3: T <- keep_largest_connected_component(T)", silent)
    tunnel_mask = keep_largest_connected_component_skimage(tunnel_mask, background=False)
    message_s(f"Step 4: Check that |T| > {tunnel_min_size}", silent)
    if np.sum(tunnel_mask) < tunnel_min_size:
        message_s(f"|T| < {tunnel_min_size} => No tunnel detected", silent)
        return np.zeros_like(tunnel_mask).astype(int)
    else:
        message_s(f"|T| >= {tunnel_min_size} => Tunnel detected", silent)
        return tunnel_mask.astype(int)


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
        args.bone_fill_gaps_radius,
        args.bone_remove_islands_radius,
        args.trab_fill_gaps_radius,
        args.silent
    )

    if args.detect_tunnel:
        message_s("Detecting tunnel...", args.silent)
        tunnel_mask = segment_tunnel(
            post_subchondral_bone_plate_mask,
            post_trabecular_bone_mask,
            args.tunnel_min_size,
            silent=args.silent
        )
    else:
        tunnel_mask = np.zeros_like(post_subchondral_bone_plate_mask)

    post_model_mask = (
            args.output_subchondral_bone_plate_class * post_subchondral_bone_plate_mask
            + args.output_trabecular_bone_class * post_trabecular_bone_mask
            + args.output_tunnel_class * tunnel_mask
    )

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
                    'supplied to this script will be saved to {output_dir}/{output_label}_postprocessed_mask.yaml.'
                    'Optionally, you can tell this script to try to autodetect an ACLR tunnel using the final '
                    'cortical and trabecular masks. In this case you will want to set the optional parameter '
                    '{tunnel-min-size} to the minimum number of voxels you expect a tunnel to occupy.',
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
        "--output-subchondral-bone-plate-class", "-osbpc", type=int, default=1, metavar="N",
        help="the class label for the subchondral bone plate in the output mask"
    )
    parser.add_argument(
        "--output-trabecular-bone-class", "-otbc", type=int, default=2, metavar="N",
        help="the class label for the trabecular bone in the output mask"
    )
    parser.add_argument(
        "--output-tunnel-class", "-otnc", type=int, default=3, metavar="N",
        help="the class label for the tunnel in the output mask"
    )
    parser.add_argument(
        "--bone-fill-gaps-radius", "-bfgr", type=int, default=5, metavar="N",
        help="radius of structural element when performing fill_gaps on the bone mask"
    )
    parser.add_argument(
        "--bone-remove-islands-radius", "-brir", type=int, default=2, metavar="N",
        help="radius of structural element when performing remove_islands on the bone mask"
    )
    parser.add_argument(
        "--minimum-subchondral-bone-plate-thickness", "-msbpt", type=int, default=4, metavar="N",
        help="minimum thickness of the subchondral bone plate, in voxels"
    )
    parser.add_argument(
        "--trab-fill-gaps-radius", "-tfgr", type=int, default=5, metavar="N",
        help="radius of structural element when performing fill_gaps on the trabecular bone mask"
    )
    parser.add_argument(
        "--detect-tunnel", "-t", action="store_true", help="try to detect ACLR tunnel"
    )
    parser.add_argument(
        "--tunnel-min-size", "-tms", type=int, default=0, metavar="N",
        help="minimum number of voxels a tunnel must occupy to be detected"
    )
    parser.add_argument("--overwrite", "-ow", action="store_true", help="Overwrite output files if they exist.")
    parser.add_argument("--silent", "-s", action="store_true", help="Silence all terminal output.")
    return parser


def main():
    args = create_parser().parse_args()
    postprocess_segmentation(args)


if __name__ == "__main__":
    main()
