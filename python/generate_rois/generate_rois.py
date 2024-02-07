from __future__ import annotations

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace

from bonelab.util.echo_arguments import echo_arguments
from bonelab.util.registration_util import check_inputs_exist, check_for_output_overwrite, message_s

import numpy as np
import SimpleITK as sitk
import yaml
import os
from tqdm import tqdm, trange
from skimage.morphology import binary_dilation, binary_erosion, binary_closing
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


def get_regional_subchondral_bone_plate_mask(
        subchondral_bone_plate_mask: np.ndarray,
        roi_mask: np.ndarray,
        roi_smoothing_sigma: float,
        regional_subchondral_bone_plate_dilation_footprint: int,
        silent: bool
) -> np.ndarray:
    if not isinstance(subchondral_bone_plate_mask, np.ndarray):
        raise ValueError("`subchondral_bone_plate_mask` must be a numpy array")
    if not isinstance(roi_mask, np.ndarray):
        raise ValueError("`roi_mask` must be a numpy array")
    message_s("Smooth out the roi_mask...", silent)
    roi_mask = gaussian(roi_mask, sigma=roi_smoothing_sigma) > 0.5
    message_s("Find the largest component of the intersection of the roi mask and subchondral bone...", silent)
    regional_subchondral_bone_plate_mask = keep_largest_connected_component_skimage(
        subchondral_bone_plate_mask & roi_mask,
        background=False
    )
    message_s("Axially dilate the regional subchondral bone plate mask and find the intersection with subchondral bone...", silent)
    regional_subchondral_bone_plate_mask = (
        binary_dilation(
            regional_subchondral_bone_plate_mask,
            np.ones((2 * regional_subchondral_bone_plate_dilation_footprint + 1, 1, 1))
        )
        & subchondral_bone_plate_mask
    ).astype(int)
    message_s("Keep only the largest connected component...", silent)
    return keep_largest_connected_component_skimage(
        regional_subchondral_bone_plate_mask,
        background=False
    )


def get_bounding_box_limits(arr: np.ndarray) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    xmin, xmax = np.where(np.any(arr, axis=(1, 2)))[0][[0, -1]]
    ymin, ymax = np.where(np.any(arr, axis=(0, 2)))[0][[0, -1]]
    zmin, zmax = np.where(np.any(arr, axis=(0, 1)))[0][[0, -1]]
    return (xmin, ymin, zmin), (xmax, ymax, zmax)


def reinsert_submask_into_full_image(
        mask: np.ndarray,
        bounds_min: Tuple[int, int, int],
        bounds_max: Tuple[int, int, int],
        original_shape: Tuple[int, int, int]
) -> np.ndarray:
    big_mask = np.zeros(original_shape, dtype=mask.dtype)
    big_mask[
        bounds_min[0]:bounds_max[0],
        bounds_min[1]:bounds_max[1],
        bounds_min[2]:bounds_max[2]
    ] = mask
    return big_mask


def generate_periarticular_rois_from_bone_plate_and_trabecular_masks(
        subchondral_bone_plate_mask: np.ndarray,
        trabecular_bone_mask: np.ndarray,
        dilation_kernel_up_single: np.ndarray,
        dilation_kernel_down_compartment: np.ndarray,
        silent: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # get the original shape of the image
    original_shape = subchondral_bone_plate_mask.shape
    # get the bounds of the bone voxels
    bounds_min, bounds_max = get_bounding_box_limits(subchondral_bone_plate_mask)
    # pad out what we consider as the bounds to include all of the space required to include the dilations
    pad_amounts = [3 * (ds - 1) // 2 for ds in dilation_kernel_down_compartment.shape]
    bounds_min = [
        max(0, bm - pa - 1)
        for bm, pa in zip(bounds_min, pad_amounts)
    ]
    bounds_max = [
        min(s, bm + pa + 1)
        for s, bm, pa in zip(subchondral_bone_plate_mask.shape, bounds_max, pad_amounts)
    ]
    # then get the subset of the original masks
    subchondral_bone_plate_mask = subchondral_bone_plate_mask[
        bounds_min[0]:bounds_max[0],
        bounds_min[1]:bounds_max[1],
        bounds_min[2]:bounds_max[2]
    ]
    trabecular_bone_mask = trabecular_bone_mask[
        bounds_min[0]:bounds_max[0],
        bounds_min[1]:bounds_max[1],
        bounds_min[2]:bounds_max[2]
    ]
    # find the top layer of bone
    message_s("Finding top layer of bone...", silent)
    top_layer_mask = (
        binary_dilation(subchondral_bone_plate_mask, dilation_kernel_up_single)
        & ~subchondral_bone_plate_mask & ~trabecular_bone_mask
    ).astype(int)
    # dilate down into the bone to get the shallow mask
    message_s("Dilating down into the bone to get the shallow mask...", silent)
    shallow_mask = top_layer_mask
    shallow_mask = binary_dilation(shallow_mask, dilation_kernel_down_compartment)
    shallow_mask = (
        shallow_mask & trabecular_bone_mask
        & ~subchondral_bone_plate_mask & ~top_layer_mask
    ).astype(int)
    # dilate down into the bone to get the mid mask
    message_s("Dilating down into the bone to get the mid mask...", silent)
    mid_mask = top_layer_mask
    for _ in trange(2, disable=silent):
        mid_mask = binary_dilation(mid_mask, dilation_kernel_down_compartment)
    mid_mask = (
        mid_mask & trabecular_bone_mask
        & ~shallow_mask & ~subchondral_bone_plate_mask & ~top_layer_mask
    ).astype(int)
    # dilate down into the bone to get the deep mask
    message_s("Dilating down into the bone to get the deep mask...", silent)
    deep_mask = top_layer_mask
    for _ in trange(3, disable=silent):
        deep_mask = binary_dilation(deep_mask, dilation_kernel_down_compartment)
    deep_mask = (
        deep_mask & trabecular_bone_mask
        & ~mid_mask & ~shallow_mask & ~subchondral_bone_plate_mask & ~top_layer_mask
    ).astype(int)
    return (
        reinsert_submask_into_full_image(
            subchondral_bone_plate_mask,
            bounds_min,
            bounds_max,
            original_shape
        ),
        reinsert_submask_into_full_image(
            shallow_mask,
            bounds_min,
            bounds_max,
            original_shape
        ),
        reinsert_submask_into_full_image(
            mid_mask,
            bounds_min,
            bounds_max,
            original_shape
        ),
        reinsert_submask_into_full_image(
            deep_mask,
            bounds_min,
            bounds_max,
            original_shape
        )
    )


def generate_rois(args: Namespace):
    print(echo_arguments("ROI Generation", vars(args)))
    # check inputs exist
    check_inputs_exist(
        [args.mask, args.atlas_mask],
        args.silent
    )
    # generate filenames for outputs
    yaml_fn = os.path.join(args.output_dir, f"{args.output_label}_roi_generation.yaml")
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
        [yaml_fn, allrois_mask_fn] + medial_roi_mask_fns + lateral_roi_mask_fns,
        args.overwrite, args.silent
    )
    # write yaml
    message_s("Writing yaml...", args.silent)
    with open(yaml_fn, "w") as f:
        yaml.dump(vars(args), f)
    # read in the mask and the atlas mask
    message_s("Reading in mask and atlas mask...", args.silent)
    mask_sitk = sitk.ReadImage(args.mask)
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
    # convert the mask and atlas mask to numpy arrays
    message_s("Converting mask and atlas mask to numpy arrays...", args.silent)
    mask = sitk.GetArrayFromImage(mask_sitk)
    atlas_mask = sitk.GetArrayFromImage(atlas_mask_sitk)
    message_s("Extract subchondral bone plate, trabecular, and tunnel masks from mask...", args.silent)
    subchondral_bone_plate_mask = (mask == args.subchondral_bone_plate_class).astype(int)
    trabecular_bone_mask = (mask == args.trabecular_bone_class).astype(int)
    tunnel_mask = (mask == args.tunnel_class).astype(int)
    message_s(f"Dilating the tunnel mask with a radius of {args.tunnel_dilation_footprint}...", args.silent)
    tunnel_mask = efficient_3d_dilation(tunnel_mask, args.tunnel_dilation_footprint)
    message_s("Creating dilation kernels...", args.silent)
    dilation_kernel_down = np.zeros((2 * args.compartment_depth + 1, 1, 1), dtype=int)
    dilation_kernel_up = np.zeros((3, 1, 1), dtype=int)
    dilation_kernel_up[1, 0, 0] = 1
    if args.bone == "femur":
        dilation_kernel_up[0, 0, 0] = 1
        dilation_kernel_down[(args.compartment_depth + 1):, 0, 0] = 1
    elif args.bone == "tibia":
        dilation_kernel_up[2, 0, 0] = 1
        dilation_kernel_down[:(args.compartment_depth + 1), 0, 0] = 1
    else:
        raise ValueError(f"bone must be `femur` or `tibia`, given {args.bone}")
    message_s("Generating medial subchondral bone plate mask...", args.silent)
    medial_subchondral_bone_plate_mask = get_regional_subchondral_bone_plate_mask(
        subchondral_bone_plate_mask,
        (atlas_mask == args.medial_atlas_code) & (~tunnel_mask),
        args.roi_smoothing_sigma,
        args.regional_subchondral_bone_plate_dilation_footprint,
        args.silent
    )
    message_s("Generating medial ROIs...", args.silent)
    medial_roi_masks = generate_periarticular_rois_from_bone_plate_and_trabecular_masks(
        medial_subchondral_bone_plate_mask,
        trabecular_bone_mask & (~tunnel_mask),
        dilation_kernel_up,
        dilation_kernel_down,
        args.silent
    )
    message_s("Generating lateral subchondral bone plate mask...", args.silent)
    lateral_subchondral_bone_plate_mask = get_regional_subchondral_bone_plate_mask(
        subchondral_bone_plate_mask,
        (atlas_mask == args.lateral_atlas_code) & (~tunnel_mask),
        args.roi_smoothing_sigma,
        args.regional_subchondral_bone_plate_dilation_footprint,
        args.silent
    )
    message_s("Generating lateral ROIs...", args.silent)
    lateral_roi_masks = generate_periarticular_rois_from_bone_plate_and_trabecular_masks(
        lateral_subchondral_bone_plate_mask,
        trabecular_bone_mask & (~tunnel_mask),
        dilation_kernel_up,
        dilation_kernel_down,
        args.silent
    )
    all_rois_mask = sitk.Image(*mask_sitk.GetSize(), mask_sitk.GetPixelID())
    all_rois_mask = sitk.Cast(all_rois_mask, sitk.sitkInt32)
    all_rois_mask.CopyInformation(mask_sitk)
    message_s("Writing medial ROI masks...", args.silent)
    for mask, fn, msc in zip(medial_roi_masks, medial_roi_mask_fns, medial_site_codes):
        mask_sitk = sitk.GetImageFromArray(mask)
        mask_sitk.CopyInformation(atlas_mask_sitk)
        sitk.WriteImage(sitk.Cast(mask_sitk, sitk.sitkInt32), fn)
        all_rois_mask += msc * sitk.Cast(mask_sitk, sitk.sitkInt32)
    message_s("Writing lateral ROI masks...", args.silent)
    for mask, fn, msc in zip(lateral_roi_masks, lateral_roi_mask_fns, lateral_site_codes):
        mask_sitk = sitk.GetImageFromArray(mask)
        mask_sitk.CopyInformation(atlas_mask_sitk)
        sitk.WriteImage(sitk.Cast(mask_sitk, sitk.sitkInt32), fn)
        all_rois_mask += msc * sitk.Cast(mask_sitk, sitk.sitkInt32)
    message_s("Writing all ROIs mask...", args.silent)
    sitk.WriteImage(all_rois_mask, allrois_mask_fn)


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description='This script takes a mask file, a bone specification (femur/tibia), and an '
                    'atlas-derived contact surface ROI mask and generates the full set of ROI masks needed to perform '
                    'the periarticular microarchitectural analysis. The output is a set of masks, one for each ROI, '
                    'and a yaml file containing the parameters used to generate the masks.'
                    'Outputs will be mask files saved to {output_dir} with filenames of the format: '
                    '{output_label}_roi{site_code}_mask.nii.gz, where site_code is the site code for the ROI. The '
                    'output yaml file will be saved to {output_dir} with filename {output_label}_roi_generation.yaml. '
                    'An additional mask, {output_label}_allrois_mask.nii.gz, will be generated that contains all of '
                    'the ROIs in a single mask, with each ROI labelled with its site code, for visualization purposes.',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("mask", type=str, help="The input mask file.")
    parser.add_argument("bone", choices=["femur", "tibia"], help="The bone to generate ROIs for.")
    parser.add_argument("atlas_mask", type=str, help="The atlas mask nii file.")
    parser.add_argument("output_dir", type=str, help="The output directory.")
    parser.add_argument("output_label", type=str, help="The output label.")
    parser.add_argument(
        "--axial-dilation-footprint", "-adf", default=20, type=int, metavar="N",
        help="the footprint to use for the axial dilation of the atlas mask"
    )
    parser.add_argument(
        "--subchondral-bone-plate-class", "-msbp", type=int, default=1, metavar="N",
        help="the class label for the subchondral bone plate in the mask. Leave this as the default if the input mask "
             "was produced by python/postprocessing/postprocess_segmentation.py"
    )
    parser.add_argument(
        "--trabecular-bone-class", "-mtb", type=int, default=2, metavar="N",
        help="the class label for the trabecular bone in the mask. Leave this as the default if the input mask "
             "was produced by python/postprocessing/postprocess_segmentation.py"
    )
    parser.add_argument(
        "--tunnel-class", "-mt", type=int, default=3, metavar="N",
        help="the class label for the tunnel in the mask. Leave this as the default if the input mask "
             "was produced by python/postprocessing/postprocess_segmentation.py"
    )
    parser.add_argument(
        "--femur-lateral-site-codes", "-flsc", default=[16, 10, 11, 12], type=int, nargs=4, metavar="N",
        help="the site codes for the lateral ROIs in the femur, the order should be: "
             "bone plate, shallow, mid, deep. Leave these as default to ensure compatibility with the "
             "periarticular microarchitectural analysis IPL scripts written by Andres Kroker"
    )
    parser.add_argument(
        "--femur-medial-site-codes", "-fmsc", default=[17, 13, 14, 15], type=int, nargs=4, metavar="N",
        help="the site codes for the medial ROIs in the femur, the order should be: "
             "bone plate, shallow, mid, deep. Leave these as default to ensure compatibility with the "
             "periarticular microarchitectural analysis IPL scripts written by Andres Kroker"
    )
    parser.add_argument(
        "--tibia-lateral-site-codes", "-tlsc", default=[36, 30, 31, 32], type=int, nargs=4, metavar="N",
        help="the site codes for the lateral ROIs in the tibia, the order should be: "
             "bone plate, shallow, mid, deep. Leave these as default to ensure compatibility with the "
             "periarticular microarchitectural analysis IPL scripts written by Andres Kroker"
    )
    parser.add_argument(
        "--tibia-medial-site-codes", "-tmsc", default=[37, 33, 34, 35], type=int, nargs=4, metavar="N",
        help="the site codes for the medial ROIs in the tibia, the order should be: "
             "bone plate, shallow, mid, deep. Leave these as default to ensure compatibility with the "
             "periarticular microarchitectural analysis IPL scripts written by Andres Kroker"
    )
    parser.add_argument(
        "--lateral-atlas-code", "-lac", default=1, type=int, metavar="N",
        help="site code used in the atlas mask for the combined lateral VOI"
    )
    parser.add_argument(
        "--medial-atlas-code", "-mac", default=2, type=int, metavar="N",
        help="site code used in the atlas mask for the combined medial VOI"
    )
    parser.add_argument(
        "--compartment-depth", "-cd", type=int, default=41, metavar="N",
        help="depth of shallow, mid, deep compartments, in voxels. Leave this as default to perform the same "
             "analysis as established by Andres Kroker (2019)"
    )
    parser.add_argument(
        "--regional-subchondral-bone-plate-dilation-footprint", "-rsbp", type=int, default=5, metavar="N",
        help="the footprint to use for the axial dilation of the regional subchondral bone plate mask, which is "
             "performed to ensure that the regional subchondral bone plate mask contains the subchondral bone plate "
             "from the endosteal to periosteal surface, and not just little bits at the sides"
    )
    parser.add_argument(
        "--roi-smoothing-sigma", "-rss", type=float, default=1.0, metavar="N",
        help="the sigma to use for smoothing the roi mask before intersecting with the subchondral bone plate mask. "
    )
    parser.add_argument(
        "--tunnel-dilation-footprint", "-tdf", type=int, default=17, metavar="N",
        help="the footprint to use for the dilation of the tunnel mask to ensure the ROIs do not include "
             "cortical bone at the border of the tunnel"
    )
    parser.add_argument("--overwrite", "-ow", action="store_true", help="Overwrite output files if they exist.")
    parser.add_argument("--silent", "-s", action="store_true", help="Silence all terminal output.")
    return parser


def main():
    args = create_parser().parse_args()
    generate_rois(args)


if __name__ == "__main__":
    main()
