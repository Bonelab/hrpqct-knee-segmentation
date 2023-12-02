from __future__ import annotations

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace, ArgumentTypeError
import SimpleITK as sitk
import numpy as np
from skimage.morphology import binary_dilation

from bonelab.util.time_stamp import message
from bonelab.util.echo_arguments import echo_arguments
from bonelab.cli.registration import check_inputs_exist, check_for_output_overwrite


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


def message_s(m: str, s: bool):
    """
    Print a message if silent is False.

    Parameters
    ----------
    m : str
        The message to print.
    s : bool
        Whether or not to print the message.

    Returns
    -------
    None
    """
    if not s:
        message(m)


def mask_image(args: Namespace) -> None:
    print(echo_arguments("Mask image", vars(args)))
    check_inputs_exist([args.input, args.mask], args.silent)
    check_for_output_overwrite(args.output, args.overwrite, args.silent)
    message_s("Reading input image...", args.silent)
    image, mask = sitk.ReadImage(args.input), sitk.ReadImage(args.mask)
    image_array, mask_array = sitk.GetArrayFromImage(image), sitk.GetArrayFromImage(mask)
    mask_array = mask_array != args.background_class
    if args.dilate_amount > 0:
        message_s("Dilating mask...", args.silent)
        mask_array = efficient_3d_dilation(mask_array, args.dilate_amount)
    message_s("Masking image...", args.silent)
    image_array[~mask_array] = args.background_value
    message_s("Writing output image...", args.silent)
    output = sitk.GetImageFromArray(image_array)
    output.CopyInformation(image)
    sitk.WriteImage(output, args.output)


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="This tool allows you to mask an image.",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "input", type=str, help="The input image."
    )
    parser.add_argument(
        "mask", type=str, help="The mask image."
    )
    parser.add_argument(
        "output", type=str, help="The output image."
    )
    parser.add_argument(
        "--dilate-amount", "-da", type=int, default=0, metavar="N",
        help="The amount to dilate the mask by."
    )
    parser.add_argument(
        "--background-class", "-bc", type=int, default=0, metavar="N",
        help="The class to use for the background."
    )
    parser.add_argument(
        "--background-value", "-bv", type=int, default=0, metavar="N",
        help="The value to use for the background."
    )
    parser.add_argument(
        "--overwrite", "-ow", action="store_true", help="Overwrite output file if it exists."
    )
    parser.add_argument(
        "--silent", "-s", action="store_true", help="Silence all terminal output."
    )
    return parser


def main() -> None:
    args = create_parser().parse_args()
    mask_image(args)


if __name__ == "__main__":
    main()
