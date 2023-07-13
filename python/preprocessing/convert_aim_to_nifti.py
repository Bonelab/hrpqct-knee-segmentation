from __future__ import annotations

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace, ArgumentTypeError
import SimpleITK as sitk
import numpy as np
import vtkbone

from bonelab.util.time_stamp import message
from bonelab.util.vtk_util import vtkImageData_to_numpy
from bonelab.util.aim_calibration_header import get_aim_density_equation
from bonelab.cli.registration import check_inputs_exist, check_for_output_overwrite


def create_parser() -> ArgumentParser:
    """
    Create the parser for the command line tool.

    Returns
    -------
    ArgumentParser
        The parser for the command line tool.
    """
    parser = ArgumentParser(
        description="This tool allows you to convert an AIM file to a NIfTI file.",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "input", type=str, help="The input AIM file."
    )
    parser.add_argument(
        "output", type=str, help="The output NIfTI file."
    )
    parser.add_argument(
        "image_type", choices=["density", "mask"], help="The type of image to convert."
    )
    parser.add_argument(
        "--overwrite", "-ow", action="store_true", help="Overwrite output file if it exists."
    )
    parser.add_argument(
        "--silent", "-s", action="store_true", help="Silence all terminal output."
    )
    return parser


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


def convert_aim_to_nifti(args: Namespace):
    """
    Convert an AIM file to a NIfTI file.

    Parameters
    ----------
    args : Namespace
        The command line arguments.

    Returns
    -------
    None
    """
    # check inputs
    check_inputs_exist([args.input], args.silent)
    # check output
    check_for_output_overwrite([args.output], args.overwrite, args.silent)
    # read image
    message_s(f"Reading AIM file {args.input}", args.silent)
    reader = vtkbone.vtkboneAIMReader()
    reader.DataOnCellsOff()
    reader.SetFileName(args.input)
    reader.Update()
    if args.image_type == "density":
        processing_log = reader.GetProcessingLog()
        density_slope, density_intercept = get_aim_density_equation(processing_log)
        arr = density_slope * vtkImageData_to_numpy(reader.GetOutput()) + density_intercept
    elif args.image_type == "mask":
        arr = vtkImageData_to_numpy(reader.GetOutput())
    else:
        raise ArgumentTypeError(f"Image type {args.image_type} not recognized.")
    # convert to SimpleITK image
    message_s("Converting to SimpleITK image", args.silent)
    img = sitk.GetImageFromArray(np.moveaxis(arr, [0, 1, 2], [2, 1, 0]))
    img.SetSpacing(reader.GetOutput().GetSpacing())
    img.SetOrigin(reader.GetOutput().GetOrigin())
    # write image
    message_s(f"Writing NIfTI file {args.output}", args.silent)
    sitk.WriteImage(img, args.output)


def main():
    parser = create_parser()
    args = parser.parse_args()
    convert_aim_to_nifti(args)


if __name__ == "__main__":
    main()
