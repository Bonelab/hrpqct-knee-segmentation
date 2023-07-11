from __future__ import annotations

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
import SimpleITK as sitk
import vtkbone
import os

from bonelab.util.time_stamp import message
from bonelab.util.echo_arguments import echo_arguments
from bonelab.util.vtk_util import vtkImageData_to_numpy
from bonelab.util.aim_calibration_header import get_aim_density_equation
from bonelab.cli.registration import check_inputs_exist, check_for_output_overwrite, read_image, get_output_base
from bonelab.cli.demons_registration import (
    construct_multiscale_progression, create_centering_transform, add_initial_transform_to_displacement_field
)
from bonelab.util.multiscale_registration import smooth_and_resample, multiscale_demons


def create_parser() -> ArgumentParser:
    """
    Create the argument parser for the generate_periarticular_rois command line tool.

    Returns
    -------
    ArgumentParser
        The argument parser for the generate_periarticular_rois command line tool.
    """
    parser = ArgumentParser(
        description="This tool will read in a knee HRpQCT image and generate a set of ROIs that define the subchondral "
                    "bone, shallow trabecular bone, mid trabecular bone, and deep trabecular bone at the contact "
                    "surface of the femoral condyles or tibial plateaus. You must specify whether the image is of a "
                    "left or right knee, an atlas image, an atlas mask, and a yaml that contains the parameters for "
                    "deformably registering the atlas to your image. You must also specify a yaml that contains the "
                    "hyperparameters  for a deep learning model trained to segment subchondral bone plate, as well as "
                    "a trained model parameters file. Finally, you must provide a yaml file containing the parameters "
                    "for the post-processing and peri-articular ROI generation.",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("image", type=str, help="The image to generate ROIs for.")
    parser.add_argument("side", type=str, choices=["L", "R"], help="Is it a left knee or a right knee.")
    parser.add_argument("atlas", type=str, help="The atlas image to use.")
    parser.add_argument("atlas_mask", type=str, help="The atlas mask to use.")
    parser.add_argument("deformable_registration_parameters", type=str,
                        help="The yaml file containing the parameters for the deformable registration.")
    parser.add_argument("deep_learning_hyperparameters", type=str,
                        help="The yaml file containing the hyperparameters for initializing the deep learning model.")
    parser.add_argument("deep_learning_model", type=str, help="The trained deep learning model file.")
    parser.add_argument("post_processing_parameters", type=str, help="The yaml file containing the parameters for "
                                                                     "post-processing.")
    parser.add_argument("output", type=str, help="The output directory to write the ROIs to.")
    parser.add_argument("--silent", "-s", action="store_true", help="Silence all terminal output.")
    parser.add_argument("--overwrite", "-ow", action="store_true", help="Overwrite any existing files.")

    return parser


def message_s(m: str, s: bool):
    """

    Parameters
    ----------
    m : str
        The message to print.
    s : bool
        Suppress messages to the terminal.

    Returns
    -------
    None.
    """
    if not s:
        message(m)


def read_aim(fn: str) -> Tuple[vtkbone.vtkboneAIMReader, sitk.Image]:
    """
    Read in an AIM file using vtkbone and convert it to a SimpleITK image.

    Parameters
    ----------
    fn : str
        The filename of the AIM file to read in.

    Returns
    -------
    reader : vtkbone.vtkboneAIMReader
        The vtkbone reader used to read in the AIM file.

    image : sitk.Image
        The SimpleITK image converted from the AIM file.
    """

    reader = vtkbone.vtkboneAIMReader()
    reader.DataOnCellsOff()
    reader.SetFileName(fn)
    reader.Update()
    processing_log = reader.GetProcessingLog()
    density_slope, density_intercept = get_aim_density_equation(processing_log)
    image = sitk.GetImageFromArray(density_slope*vtkImageData_to_numpy(reader.GetOutput()) + density_intercept)
    image.SetSpacing(reader.GetOutput().GetSpacing())
    image.SetOrigin(reader.GetOutput().GetOrigin())
    return reader, image


def get_image_center(img: sitk.Image) -> Tuple[float, ...]:
    """
    Get the center of the image in physical coordinates.

    Parameters
    ----------
    img : sitk.Image
        The image to get the center of.

    Returns
    -------
    Tuple[float, ...]
        The center of the image in physical coordinates.
    """
    origin = np.asarray(img.GetOrigin())
    spacing = np.asarray(img.GetSpacing())
    size = np.asarray(img.GetSize())
    direction = np.asarray(img.GetDirection()).reshape(3, 3)
    return origin + np.matmul(direction, (spacing*size/2))


def mirror_image(img: sitk.Image, label: str, axis: int, silent: bool) -> sitk.Image:
    """
    Mirror the given image on the given axis.

    Parameters
    ----------
    img : sitk.Image
        The image to mirror.
    label : str
        The label of the image to mirror.
    axis : int
        The axis to mirror the image on.
    silent : bool
        Suppress messages to the terminal.

    Returns
    -------
    sitk.Image
        The mirrored image.
    """
    # create an affine transform
    message_s(f"Creating transform that will mirror {label} on axis {axis}", silent)
    transform = sitk.AffineTransform(img.GetDimension())
    transform.SetCenter(get_image_center(img))
    transform.Scale([-1 if i == axis else 1 for i in range(img.GetDimension())])
    message(f"Mirroring {label}", silent)
    return sitk.Resample(img, img, transform, sitk.sitkLinear)


def register_atlas_to_image_and_transform_atlas_mask(
        image: sitk.Image, atlas: sitk.Image, atlas_mask: sitk.Image, parameters: dict, silent: bool
) -> sitk.Image:
    """
    Register the atlas to the image and transform the atlas mask.

    Parameters
    ----------
    image : sitk.Image
        The image to register the atlas to.

    atlas : sitk.Image
        The atlas to register to the image.

    atlas_mask : sitk.Image
        The atlas mask to transform.

    parameters : dict
        The parameters for the registration.

    silent : bool
        Suppress messages to the terminal.

    Returns
    -------

    """
    pass


def generate_crude_bone_mask(image: sitk.Image, parameters: dict) -> sitk.Image:
    """
    Generate a crude bone mask from the given image.

    Parameters
    ----------
    image : sitk.Image
        The image to generate the crude bone mask from.
    parameters : dict
        The parameters for generating the crude bone mask.

    Returns
    -------
    sitk.Image
        The crude bone mask.
    """
    # TODO: Implement this
    return np.zeros(image.shape, dtype=np.bool)


def generate_periarticular_rois(args: Namespace):
    """
    Generate the peri-articular ROIs for the given image.

    Parameters
    ----------
    args : Namespace
        The arguments parsed from the command line.

    Returns
    -------
    None.
    """
    print(echo_arguments("Peri-articular ROI Generation Script", vars(args)))
    # 0. Error checking
    # 0.1 Check that all inputs exist
    check_inputs_exist(
        [
            args.image, args.atlas, args.atlas_mask, args.deformable_registration_parameters,
            args.deep_learning_parameters, args.deep_learning_model, args.post_processing_parameters
        ],
        args.silent
    )
    # 0.2 Check that the output directory exists, if not then create it
    if not os.path.exists(args.output):
        message_s(f"Creating output directory: {args.output}", args.silent)
        os.makedirs(args.output)
    # 0.3 Check if the outputs already exist, if so then exit or overwrite
    check_for_output_overwrite(args.output, args.overwrite, args.silent)
    # 1. Read in the image using vtkbone, and convert to density
    reader, image = read_aim(args.image)
    # 2. Read in the atlas using SimpleITK
    atlas = sitk.ReadImage(args.atlas)
    # 3. Read in the atlas mask using SimpleITK
    atlas_mask = sitk.ReadImage(args.atlas_mask)
    # 4. Read in the deformable registration parameters yaml
    with open(args.deformable_registration_parameters, 'r') as f:
        deformable_registration_parameters = yaml.load(f, Loader=yaml.FullLoader)
    # 5. Register the atlas to the image using the deformable registration parameters and transform mask to image
    # - If the knee is a left knee, then the atlas, and mask, should be flipped in the x-direction before registration
    if args.side == "L":
        message_s("Left knee, flipping atlas and atlas mask in the x-direction.", args.silent)
        atlas = mirror_image(atlas, "atlas", 0, args.silent)
        atlas_mask = mirror_image(atlas_mask, "atlas mask", 0, args.silent)
    elif args.side == "R":
        message_s("Right knee, not flipping atlas and atlas mask in the x-direction.", args.silent)
    else:
        raise ValueError("`side` must be either 'L' or 'R'.")
    transformed_atlas_mask = register_atlas_to_image_and_transform_atlas_mask(
        image, atlas, atlas_mask, deformable_registration_parameters, args.silent
    )
    # 6. Read in the post-processing parameters yaml
    with open(args.post_processing_parameters, 'r') as f:
        post_processing_parameters = yaml.load(f, Loader=yaml.FullLoader)
    # 7. Generate a crude bone mask from the image
    bone_mask = generate_crude_bone_mask(image, post_processing_parameters)
    # 8. Erode the crude bone mask in the frontal (L/R) direction
    # 9. Find the intersection of the medial/lateral ROI masks and the eroded bone mask
    # 10. Extract a volume from the image using the intersection masks
    # 11. Read in the deep learning parameters
    # 12. Load the deep learning model
    # 13. Run the deep learning model on the extracted volumes to get the subchondral bone plate masks
    # 14. Generate the shallow, mid, and deep trabecular bone masks by translating the periosteal surface of the
    #     subchondral bone plate masks into the bone specific distances, as defined in the post-processing parameters
    # 15. Write the ROIs to disk as AIM files, carrying over the processing log from the original image but adding
    #     a new entry for the peri-articular ROI generation


def main():
    args = create_parser().parse_args()
    generate_periarticular_rois(args)


if __name__ == '__main__':
    main()
