from __future__ import annotations

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
import SimpleITK as sitk
import numpy as np
import math
from scipy import stats
from typing import Tuple, List

from bonelab.util.echo_arguments import echo_arguments
from bonelab.util.registration_util import (
    read_image, check_inputs_exist, check_for_output_overwrite, write_args_to_yaml,
    create_file_extension_checker, create_string_argument_checker,
    INPUT_EXTENSIONS, get_output_base, write_metrics_to_csv, create_and_save_metrics_plot,
    setup_optimizer, setup_interpolator, setup_similarity_metric, setup_multiscale_progression,
    check_percentage, check_image_size_and_shrink_factors, INTERPOLATORS
)
from bonelab.util.demons_registration_util import (
    multiscale_demons, smooth_and_resample,
    IMAGE_EXTENSIONS, DEMONS_FILTERS, demons_type_checker, construct_multiscale_progression
)
from bonelab.util.time_stamp import message


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Affine Atlas Generation Script",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--images", "-img",
        type=create_file_extension_checker(INPUT_EXTENSIONS, "images"), nargs="+",
        metavar="IMAGES",
        help=f"Provide image filenames ({', '.join(INPUT_EXTENSIONS)}).",
        required=True
    )
    parser.add_argument(
        "--masks", "-msk",
        type=create_file_extension_checker(INPUT_EXTENSIONS, "masks"), nargs="+",
        metavar="MASKS",
        help=f"Provide image filenames ({', '.join(INPUT_EXTENSIONS)}).",
        required=True
    )
    parser.add_argument(
        "atlas_average",
        type=create_file_extension_checker(IMAGE_EXTENSIONS, "atlas_average"),
        metavar="ATLAS_AVERAGE",
        help=f"Provide output filename for the average-atlas image ({', '.join(IMAGE_EXTENSIONS)})."
    )
    parser.add_argument(
        "atlas_mask",
        type=create_file_extension_checker(IMAGE_EXTENSIONS, "atlas_average"),
        metavar="ATLAS_MASK",
        help=f"Provide output filename for the average-atlas image ({', '.join(IMAGE_EXTENSIONS)})."
    )
    parser.add_argument(
        "--overwrite", "-ow", default=False, action="store_true",
        help="enable this flag to overwrite existing files, if they exist at output targets"
    )
    parser.add_argument(
        "--downsampling-shrink-factor", "-dsf", default=None, type=int, metavar="X",
        help="factor by which to shrink every image when it is first read"
    )
    parser.add_argument(
        "--downsampling-smoothing-sigma", "-dss", default=None, type=float, metavar="X",
        help="variance for the Gaussian filter used to smooth every image when it is first read"
    )
    parser.add_argument(
        "--pad-amount", "-pa", default=20, type=int, metavar="X",
        help="how much to pad images by after downsampling"
    )
    parser.add_argument(
        "--background-value", "-bv", default=-400, type=int, metavar="X",
        help="default value to assign to voxels during padding, resampling"
    )
    parser.add_argument(
        "--shrink-factors", "-sf", default=None, type=int, nargs="+", metavar="X",
        help="factors by which to shrink the fixed and moving image at each stage of the multiscale progression. you "
             "must give the same number of arguments here as you do for `smoothing-sigmas`"
    )
    parser.add_argument(
        "--smoothing-sigmas", "-ss", default=None, type=float, nargs="+", metavar="X",
        help="variances for the Gaussians used to smooth the fixed and moving image at each stage of the multiscale "
             "progression. you must give the same number of arguments here as you do for `shrink-factors`"
    )
    parser.add_argument(
        "--max-affine-iterations", "-mai", default=100, type=int, metavar="N",
        help="number of iterations to run registration algorithm for at each stage in the affine registration"
    )
    parser.add_argument(
        "--optimizer", "-opt", default="GradientDescent", metavar="STR",
        type=create_string_argument_checker(["GradientDescent", "Powell"], "optimizer"),
        help="the optimizer to use, options: `GradientDescent`, `Powell`"
    )
    parser.add_argument(
        "--gradient-descent-learning-rate", "-gdlr", default=1e-3, type=float, metavar="X",
        help="learning rate when using gradient descent optimizer"
    )
    parser.add_argument(
        "--gradient-descent-convergence-min-value", "-gdcmv", default=1e-6, type=float, metavar="X",
        help="minimum value for convergence when using gradient descent optimizer"
    )
    parser.add_argument(
        "--gradient-descent-convergence-window-size", "-gdcws", default=10, type=int, metavar="N",
        help="window size for checking for convergence when using gradient descent optimizer"
    )
    parser.add_argument(
        "--powell_max_line_iterations", "-pmli", default=100, type=int, metavar="N",
        help="maximum number of line iterations when using Powell optimizer"
    )
    parser.add_argument(
        "--powell_step_length", "-psl", default=1.0, type=float, metavar="X",
        help="maximum step length when using Powell optimizer"
    )
    parser.add_argument(
        "--powell_step_tolerance", "-pst", default=1e-6, type=float, metavar="X",
        help="step tolerance when using Powell optimizer"
    )
    parser.add_argument(
        "--powell_value_tolerance", "-pvt", default=1e-6, type=float, metavar="X",
        help="value tolerance when using Powell optimizer"
    )
    parser.add_argument(
        "--similarity-metric", "-sm", default="MeanSquares", metavar="STR",
        type=create_string_argument_checker(
            ["MeanSquares", "Correlation", "JointHistogramMutualInformation", "MattesMutualInformation"],
            "similarity-metric"
        ),
        help="the similarity metric to use, options: `MeanSquares`, `Correlation`, "
             "`JointHistogramMutualInformation`, `MattesMutualInformation`"
    )
    parser.add_argument(
        "--similarity-metric-sampling-strategy", "-smss", default="None", metavar="STR",
        type=create_string_argument_checker(["None", "Regular", "Random"], "similarity-metric-sampling-strategy"),
        help="sampling strategy for similarity metric, options: "
             "`None` -> use all points, "
             "`Regular` -> sample on a regular grid with specified sampling rate, "
             "`Random` -> sample randomly with specified sampling rate."
    )
    parser.add_argument(
        "--similarity-metric-sampling-rate", "-smsr", default=0.2, type=check_percentage, metavar="P",
        help="sampling rate for similarity metric, must be between 0.0 and 1.0"
    )
    parser.add_argument(
        "--similarity-metric-sampling-seed", "-smssd", default=None, type=int, metavar="N",
        help="the seed for random sampling, leave as `None` if you want a random seed. Can be useful if you want a "
             "deterministic registration with random sampling for debugging/testing. Don't go crazy and use huge "
             "numbers since SITK might report an OverflowError. I found keeping it <=255 worked."
    )
    parser.add_argument(
        "--mutual-information-num-histogram-bins", "-minhb", default=20, type=int, metavar="N",
        help="number of bins in histogram when using joint histogram or Mattes mutual information similarity metrics"
    )
    parser.add_argument(
        "--joint-mutual-information-joint-smoothing-variance", "-jmijsv", default=1.5, type=float, metavar="X",
        help="variance to use when smoothing the joint PDF when using the joint histogram mutual information "
             "similarity metric"
    )
    parser.add_argument(
        "--interpolator", "-int", default="Linear", metavar="STR",
        type=create_string_argument_checker(list(INTERPOLATORS.keys()), "interpolator"),
        help="the interpolator to use, options: `Linear`, `NearestNeighbour`, `BSpline`"
    )
    parser.add_argument(
        "--max-demons-iterations", "-mdi", default=100, type=int, metavar="N",
        help="number of iterations to run registration algorithm for at each stage in the demons registration"
    )
    parser.add_argument(
        "--demons-type", "-dt", default="demons", type=demons_type_checker, metavar="STR",
        help=f"type of demons algorithm to use. options: {list(DEMONS_FILTERS.keys())}"
    )
    parser.add_argument(
        "--displacement-smoothing-std", "-ds", default=1.0, type=float, metavar="X",
        help="standard deviation for the Gaussian smoothing applied to the displacement field at each step."
             "this is how you control the elasticity of the smoothing of the deformation"
    )
    parser.add_argument(
        "--update-smoothing-std", "-us", default=1.0, type=float, metavar="X",
        help="standard deviation for the Gaussian smoothing applied to the update field at each step."
             "this is how you control the viscosity of the smoothing of the deformation"
    )
    parser.add_argument(
        "--lateral-site-codes", "-lsc", default=[10, 11, 12, 16, 30, 31, 32, 36], type=int, nargs="+", metavar="N",
        help="the site codes for the lateral VOIs"
    )
    parser.add_argument(
        "--medial-site-codes", "-msc", default=[13, 14, 15, 17, 33, 34, 35, 37], type=int, nargs="+", metavar="N",
        help="the site codes for the medial VOIs"
    )
    parser.add_argument(
        "--lateral-output-code", "-loc", default=1, type=int, metavar="N",
        help="site code to use in final atlas mask for the combined lateral VOI"
    )
    parser.add_argument(
        "--medial-output-code", "-moc", default=2, type=int, metavar="N",
        help="site code to use in final atlas mask for the combined medial VOI"
    )
    parser.add_argument(
        "--STAPLE-binarization-threshold", "-Sbt", type=float, default=0.5, metavar="X",
        help="threshold to use to binarize STAPLE output masks"
    )
    parser.add_argument(
        "--silent", "-s", default=False, action="store_true",
        help="enable this flag to suppress terminal output about how the registration is proceeding"
    )

    return parser


def message_s(m: str, s: bool):
    if not s:
        message(m)


def read_and_downsample_image(
        image: str,
        label: str,
        downsampling_shrink_factor: float,
        downsampling_smoothing_sigma: float,
        pad_amount: int,
        pad_value: float,
        silent: bool
) -> sitk.Image:
    """
    Read an image and downsample it if requested.

    Parameters
    ----------
    image : str
        Filename of image to read.

    label : str
        Label for image, used for terminal output.

    downsampling_shrink_factor : float
        Downsampling factor.

    downsampling_smoothing_sigma : float
        Downsampling smoothing sigma.

    pad_amount : int
        Amount to pad image by on all sides.

    pad_value : float
        The value to use for padding.

    silent : bool
        Suppress terminal output.

    Returns
    -------
    sitk.Image

    """
    # load images, cast to single precision float
    image = sitk.Cast(read_image(image, label, silent), sitk.sitkFloat32)
    # optionally, downsample the fixed and moving images
    if (downsampling_shrink_factor is not None) and (downsampling_smoothing_sigma is not None):
        message_s(f"Downsampling and smoothing {label} with shrink factor {downsampling_shrink_factor} and sigma "
                  f"{downsampling_smoothing_sigma}.", silent)
        image = smooth_and_resample(
            image, downsampling_shrink_factor, downsampling_smoothing_sigma
        )
    elif (downsampling_shrink_factor is None) and (downsampling_smoothing_sigma is None):
        # do not downsample fixed and moving images
        message_s(f"Using {label} at full resolution.", silent)
    else:
        raise ValueError("one of `downsampling-shrink-factor` or `downsampling-smoothing-sigma` have not been specified"
                         " - you must either leave both as the default `None` or specify both")
    message_s(f"Padding {label} by {pad_amount}", silent)
    return sitk.ConstantPad(
        image,
        (pad_amount, pad_amount, pad_amount),
        (pad_amount, pad_amount, pad_amount),
        pad_value
    )


def affine_registration(atlas: sitk.Image, image: sitk.Image, args: Namespace) -> sitk.Transform:
    message_s("Affinely registering...", args.silent)
    registration_method = sitk.ImageRegistrationMethod()
    # hard-code to use the geometry initialization of the transform
    registration_method.SetInitialTransform(
        sitk.CenteredTransformInitializer(
            atlas, image,
            sitk.AffineTransform(atlas.GetDimension()),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
    )
    # but set up the optimizer, similarity metric, interpolator, and multiscale progression as normal using args
    registration_method = setup_optimizer(
        registration_method,
        args.max_affine_iterations,
        args.gradient_descent_learning_rate,
        args.gradient_descent_convergence_min_value,
        args.gradient_descent_convergence_window_size,
        args.powell_max_line_iterations,
        args.powell_step_length,
        args.powell_step_tolerance,
        args.powell_value_tolerance,
        args.optimizer,
        args.silent
    )
    registration_method = setup_similarity_metric(
        registration_method,
        args.similarity_metric,
        args.mutual_information_num_histogram_bins,
        args.joint_mutual_information_joint_smoothing_variance,
        args.similarity_metric_sampling_strategy,
        args.similarity_metric_sampling_rate,
        args.similarity_metric_sampling_seed,
        args.silent
    )
    registration_method = setup_interpolator(registration_method, args.interpolator, args.silent)
    message_s("Starting registration.", args.silent)
    transform = registration_method.Execute(atlas, image)
    message_s(
        f"Registration stopping condition: {registration_method.GetOptimizerStopConditionDescription()}",
        args.silent
    )
    return transform


def deformable_registration_and_masks_transformation(
        atlas: sitk.Image, image: sitk.Image, masks: Tuple[sitk.Image, sitk.Image], label: str, args: Namespace
) -> Tuple[sitk.Image, sitk.Image]:
    message_s(f"Deformably registering {label}", args.silent)
    transform = sitk.CenteredTransformInitializer(
        atlas, image,
        sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    image = sitk.Resample(image, atlas, transform, sitk.sitkLinear, defaultPixelValue=args.background_value)
    displacement, _ = multiscale_demons(
        atlas, image,
        demons_type=args.demons_type,
        demons_iterations=args.max_demons_iterations,
        demons_displacement_field_smooth_std=args.displacement_smoothing_std,
        demons_update_field_smooth_std=args.update_smoothing_std,
        initial_transform=None,
        multiscale_progression=construct_multiscale_progression(
            args.shrink_factors, args.smoothing_sigmas, args.silent
        ),
        silent=args.silent
    )
    displacement_field_transform = sitk.DisplacementFieldTransform(sitk.Add(
        displacement,
        sitk.TransformToDisplacementField(
            transform,
            displacement.GetPixelID(),
            displacement.GetSize(),
            displacement.GetOrigin(),
            displacement.GetSpacing(),
            displacement.GetDirection()
        )
    ))
    message_s(f"Transforming mask of {label}", args.silent)
    return (
        sitk.Resample(masks[0], atlas, displacement_field_transform),
        sitk.Resample(masks[1], atlas, displacement_field_transform)
    )


def get_medial_and_lateral_masks(
        mask_fn: str, label: str, medial_site_codes: List[int], lateral_site_codes: List[int], silent: bool
) -> Tuple[sitk.Image, sitk.Image]:
    mask = sitk.Cast(read_image(mask_fn, label, silent), sitk.sitkUInt8)
    medial_mask = sitk.Image(*mask.GetSize(), mask.GetPixelIDValue())
    medial_mask.CopyInformation(mask)
    for medial_site_code in medial_site_codes:
        medial_mask += (mask == medial_site_code)
    lateral_mask = sitk.Image(*mask.GetSize(), mask.GetPixelIDValue())
    lateral_mask.CopyInformation(mask)
    for lateral_site_code in lateral_site_codes:
        lateral_mask += (mask == lateral_site_code)
    return medial_mask, lateral_mask


def generate_affine_atlas(args: Namespace) -> None:
    print(echo_arguments("Generate Affine Atlas", vars(args)))
    # error checking
    output_base = get_output_base(args.atlas_average, INPUT_EXTENSIONS, args.silent)
    output_yaml = f"{output_base}.yaml"
    if len(args.images) < 2:
        raise ValueError(f"Cannot construct an average atlas with less than 2 reference images, "
                         f"given {len(args.images)}")
    if len(args.images) != len(args.masks):
        raise ValueError(f"must be same number of images and masks, got {len(args.images)} and {args.masks}")
    check_inputs_exist(args.images + args.masks, args.silent)
    check_for_output_overwrite(
        [args.atlas_average, args.atlas_mask, output_yaml],
        args.overwrite, args.silent
    )
    # write args to yaml file
    write_args_to_yaml(output_yaml, args, args.silent)
    # create the atlas
    message_s("Atlas creation starting..", args.silent)
    atlas = read_and_downsample_image(
        args.images[0], f"image 0",
        args.downsampling_shrink_factor,
        args.downsampling_smoothing_sigma,
        args.pad_amount, args.background_value,
        args.silent
    )
    average_image = sitk.Image(*atlas.GetSize(), atlas.GetPixelID())
    average_image.CopyInformation(atlas)
    average_image = sitk.Add(average_image, atlas)
    for i, img_fn in enumerate(args.images[1:]):
        message_s(f"-- Image {i+1}.", args.silent)
        image = read_and_downsample_image(
            img_fn, f"image {i+1}",
            args.downsampling_shrink_factor,
            args.downsampling_smoothing_sigma,
            args.pad_amount, args.background_value,
            args.silent
        )
        transform = affine_registration(atlas, image, args)
        message_s("Adding transformed image to average image...", args.silent)
        average_image = sitk.Add(
            average_image,
            sitk.Resample(image, atlas, transform, sitk.sitkLinear, defaultPixelValue=args.background_value)
        )
    message_s("Dividing accumulated average image by number of images...", args.silent)
    atlas = sitk.Divide(average_image, len(args.images))
    message_s(f"Saving atlas to {args.atlas_average}", args.silent)
    sitk.WriteImage(atlas, args.atlas_average)
    message_s(f"Deformably registering images and transforming masks to atlas space...", args.silent)
    transformed_masks = []
    for i, (img_fn, mask_fn) in enumerate(zip(args.images, args.masks)):
        message_s(f"-- Image {i}", args.silent)
        image = read_and_downsample_image(
            img_fn, f"image {i}",
            args.downsampling_shrink_factor,
            args.downsampling_smoothing_sigma,
            args.pad_amount, args.background_value,
            args.silent
        )
        masks = get_medial_and_lateral_masks(
            mask_fn, f"mask {i}", args.medial_site_codes, args.lateral_site_codes, args.silent
        )
        transformed_masks.append(
            deformable_registration_and_masks_transformation(atlas, image, masks, f"image {i}", args)
        )
    message_s("Initializing the atlas mask", args.silent)
    atlas_mask = sitk.Image(*atlas.GetSize(), atlas.GetPixelIDValue())
    atlas_mask.CopyInformation(atlas)
    atlas_mask = sitk.Cast(atlas_mask, sitk.sitkUInt8)
    message_s("Using STAPLE to get a consensus medial mask for the atlas", args.silent)
    atlas_mask += (
        args.medial_output_code * sitk.BinaryThreshold(
            sitk.STAPLE([masks[0] for masks in transformed_masks]), args.STAPLE_binarization_threshold, 1e6
        )
    )
    message_s("Using STAPLE to get a consensus lateral mask for the atlas", args.silent)
    atlas_mask += (
        args.lateral_output_code * sitk.BinaryThreshold(
            sitk.STAPLE([masks[1] for masks in transformed_masks]), args.STAPLE_binarization_threshold, 1e6
        )
    )
    message_s(f"Writing atlas mask to {args.atlas_mask}", args.silent)
    sitk.WriteImage(atlas_mask, args.atlas_mask)


def main() -> None:
    args = create_parser().parse_args()
    generate_affine_atlas(args)


if __name__ == "__main__":
    main()
