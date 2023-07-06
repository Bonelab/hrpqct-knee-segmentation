from __future__ import annotations

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
import SimpleITK as sitk
from typing import Tuple, List

from bonelab.util.echo_arguments import echo_arguments
from bonelab.util.time_stamp import message
from bonelab.cli.registration import read_image


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="VOI Mask Combining Script",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_mask", type=str, help="the filename of the mask with all of the peri-articular VOIs")
    parser.add_argument("output_mask", type=str, help="the filename of the combined mask")
    parser.add_argument(
        "--lateral-site-codes", "-lsc", default=[13, 14, 15, 17, 33, 34, 35, 37], type=int, nargs="+", metavar="N",
        help="the site codes for the lateral VOIs"
    )
    parser.add_argument(
        "--medial-site-codes", "-msc", default=[10, 11, 12, 16, 30, 31, 32, 36], type=int, nargs="+", metavar="N",
        help="the site codes for the lateral VOIs"
    )
    parser.add_argument(
        "--lateral-output-code", "-loc", default=2, type=int, metavar="N",
        help="site code to use in final atlas mask for the combined lateral VOI"
    )
    parser.add_argument(
        "--medial-output-code", "-moc", default=1, type=int, metavar="N",
        help="site code to use in final atlas mask for the combined medial VOI"
    )
    parser.add_argument(
        "--silent", "-s", default=False, action="store_true",
        help="enable this flag to suppress terminal output about how the registration is proceeding"
    )

    return parser


def message_s(m: str, s: bool):
    if not s:
        message(m)


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


def combine_roi_masks(args: Namespace):
    print(echo_arguments("VOI mask combining script", vars(args)))
    medial_mask, lateral_mask = get_medial_and_lateral_masks(
        args.input_mask, "input mask", args.medial_site_codes, args.lateral_site_codes, args.silent
    )
    message_s("Combining masks", args.silent)
    mask = args.lateral_output_code*lateral_mask + args.medial_output_code*medial_mask
    message_s(f"Writing output mask to {args.output_mask}", args.silent)
    sitk.WriteImage(mask, args.output_mask)


def main():
    args = create_parser().parse_args()
    combine_roi_masks(args)


if __name__ == "__main__":
    main()
