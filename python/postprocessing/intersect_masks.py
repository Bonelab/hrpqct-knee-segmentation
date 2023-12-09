from __future__ import annotations

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace

import numpy as np
from bonelab.util.echo_arguments import echo_arguments
from bonelab.util.registration_util import check_inputs_exist, check_for_output_overwrite, message_s

import SimpleITK as sitk


def intersect_masks(args: Namespace) -> None:
    if not args.silent:
        print(echo_arguments("Intersect Masks", vars(args)))

    check_inputs_exist(args.inputs, args.silent)
    check_for_output_overwrite(args.output, args.overwrite, args.silent)

    if len(args.inputs) == 1:
        if not silent:
            print(message_s("Only one input mask provided, copying to output", args.silent))
        sitk.WriteImage(sitk.ReadImage(args.inputs[0]), args.output)
        return

    message_s("Reading in masks", args.silent)
    input_masks = [sitk.ReadImage(fn) for fn in args.inputs]
    message_s("Converting masks to arrays", args.silent)
    input_arrays = [sitk.GetArrayFromImage(mask) for mask in input_masks]
    message_s("Initializing output array", args.silent)
    output_array = np.zeros_like(input_arrays[0], dtype=np.uint8)

    message_s("Beginning iteration over classes", args.silent)
    for c in args.classes:
        message_s(f"Intersecting class {c}", args.silent)
        message_s("Generating list of binary class arrays", args.silent)
        class_arrays = [(arr == c) for arr in input_arrays]
        message_s("Initializing intersected class array", args.silent)
        class_intersection = np.ones_like(class_arrays[0], dtype=np.uint8)
        message_s("Iterating over binary class arrays", args.silent)
        for arr in class_arrays:
            class_intersection = class_intersection & arr
        message_s("Adding intersected class array to output array", args.silent)
        output_array += c * class_intersection

    message_s("Converting output array to SimpleITK image", args.silent)
    output_mask = sitk.GetImageFromArray(output_array)
    output_mask.CopyInformation(input_masks[0])
    message_s("Writing output mask", args.silent)
    sitk.WriteImage(output_mask, args.output)


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Intersect masks", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--inputs", "-i", help="Input masks", nargs="+", required=True)
    parser.add_argument("--classes", "-c", help="Classes to intersect", type=int, nargs="+", required=True)
    parser.add_argument("--output", "-o", help="Output mask", required=True)
    parser.add_argument("--overwrite", "-ow", help="Overwrite output", action="store_true")
    parser.add_argument("--silent", "-s", help="Silence output", action="store_true")
    return parser


def main() -> None:
    args = create_parser().parse_args()
    intersect_masks(args)


if __name__ == '__main__':
    main()