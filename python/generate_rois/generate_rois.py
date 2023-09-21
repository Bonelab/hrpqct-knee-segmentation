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
    # find the top layer of bone
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
    for _ in trange(2*compartment_depth, disable=silent):
        mid_mask = binary_dilation(mid_mask, dilation_kernel_down)
    mid_mask = (
            mid_mask & trabecular_bone_mask
            & ~shallow_mask & ~subchondral_bone_plate_mask & ~top_layer_mask
    ).astype(int)
    # dilate down into the bone to get the deep mask
    message_s("Dilating down into the bone to get the deep mask...", silent)
    deep_mask = top_layer_mask
    for _ in trange(3*compartment_depth, disable=silent):
        deep_mask = binary_dilation(deep_mask, dilation_kernel_down)
    deep_mask = (
            deep_mask & trabecular_bone_mask
            & ~mid_mask & ~shallow_mask & ~subchondral_bone_plate_mask & ~top_layer_mask
    ).astype(int)
    return subchondral_bone_plate_mask, shallow_mask, mid_mask, deep_mask


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
    message_s("Extract subchondral bone plate and trabecular bone masks from mask...", args.silent)
    subchondral_bone_plate_mask = (mask == args.model_subchondral_bone_plate_class).astype(int)
    trabecular_bone_mask = (mask == args.model_trabecular_bone_class).astype(int)
    medial_subchondral_bone_plate_mask = get_regional_subchondral_bone_plate_mask(
        subchondral_bone_plate_mask,
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
    lateral_subchondral_bone_plate_mask = get_regional_subchondral_bone_plate_mask(
        subchondral_bone_plate_mask,
        atlas_mask == args.lateral_atlas_code,
        args.bone
    )
    lateral_trabecular_bone_mask = (
            trabecular_bone_mask
            & (atlas_mask == args.lateral_atlas_code)
    ).astype(int)
    message_s("Generating lateral ROIs...", args.silent)
    lateral_roi_masks = generate_periarticular_rois_from_bone_plate_and_trabecular_masks(
        lateral_subchondral_bone_plate_mask,
        lateral_trabecular_bone_mask,
        dilation_kernel_up,
        dilation_kernel_down,
        args.compartment_depth,
        args.silent
    )
    all_rois_mask = sitk.Image(*model_mask_sitk.GetSize(), model_mask_sitk.GetPixelID())
    all_rois_mask = sitk.Cast(all_rois_mask, sitk.sitkInt32)
    all_rois_mask.CopyInformation(model_mask_sitk)
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
        "--femur-lateral-site-codes", "-flsc", default=[17, 13, 14, 15], type=int, nargs=4, metavar="N",
        help="the site codes for the lateral ROIs in the femur, the order should be: "
             "bone plate, shallow, mid, deep. Leave these as default to ensure compatibility with the "
             "periarticular microarchitectural analysis IPL scripts written by Andres Kroker"
    )
    parser.add_argument(
        "--femur-medial-site-codes", "-fmsc", default=[16, 10, 11, 12], type=int, nargs=4, metavar="N",
        help="the site codes for the medial ROIs in the femur, the order should be: "
             "bone plate, shallow, mid, deep. Leave these as default to ensure compatibility with the "
             "periarticular microarchitectural analysis IPL scripts written by Andres Kroker"
    )
    parser.add_argument(
        "--tibia-lateral-site-codes", "-tlsc", default=[37, 33, 34, 35], type=int, nargs=4, metavar="N",
        help="the site codes for the lateral ROIs in the tibia, the order should be: "
             "bone plate, shallow, mid, deep. Leave these as default to ensure compatibility with the "
             "periarticular microarchitectural analysis IPL scripts written by Andres Kroker"
    )
    parser.add_argument(
        "--tibia-medial-site-codes", "-tmsc", default=[36, 30, 31, 32], type=int, nargs=4, metavar="N",
        help="the site codes for the medial ROIs in the tibia, the order should be: "
             "bone plate, shallow, mid, deep. Leave these as default to ensure compatibility with the "
             "periarticular microarchitectural analysis IPL scripts written by Andres Kroker"
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
        "--compartment-depth", "-cd", type=int, default=41, metavar="N",
        help="depth of shallow, mid, deep compartments, in voxels. Leave this as default to perform the same "
             "analysis as established by Andres Kroker (2019)"
    )
    parser.add_argument("--overwrite", "-ow", action="store_true", help="Overwrite output files if they exist.")
    parser.add_argument("--silent", "-s", action="store_true", help="Silence all terminal output.")
    return parser


def main():
    args = create_parser().parse_args()
    generate_rois(args)


if __name__ == "__main__":
    main()
