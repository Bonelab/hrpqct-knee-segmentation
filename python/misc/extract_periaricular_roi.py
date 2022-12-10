from __future__ import annotations

import numpy as np
import os

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from bonelab.util.aim_calibration_header import get_aim_density_equation
from bonelab.util.vtk_util import vtkImageData_to_numpy
from glob import glob
from vtkbone import vtkboneAIMReader, vtkboneAIMWriter
from enum import Enum
from pathlib import Path

# create an ImageType Enum with two options: density image or mask
ImageType = Enum("ImageType", "DENSITY MASK")


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description='HRpQCT Knee Peri-Articular ROI Extraction Script',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'image_dir', type=str, metavar='DIR',
        help='the directory where the images are stored'
    )


def get_roi_codes(bone: str):
    # for each bone/side, codes are given in a tuple in order: sc plate, shallow, medium, deep
    if bone == "F":
        return {
            "medial": (17, 13, 14, 15),
            "lateral": (16, 10, 11, 12)
        }
    elif bone == "T":
        return {
            "medial": (37, 33, 34, 35),
            "lateral": (36, 30, 31, 32)
        }
    else:
        raise ValueError("We only have codes for `F` or `T`")


def read_image_to_numpy(reader: vtkboneAIMReader, filename: str, image_type: ImageType) -> Tuple[np.ndarray, list]:
    reader.SetFileName(filename)
    reader.Update()
    data = vtkImageData_to_numpy(reader.GetOutput())
    if image_type is ImageType.DENSITY:
        # convert data to densities
        m, b = get_aim_density_equation(reader.GetProcessingLog())
        data = m * data + b
    elif image_type is ImageType.MASK:
        # convert data to a binary mask
        data = (data > 0).astype(int)
    return data, reader.GetPosition()


def align_aims(data: List[Tuple[np.ndarray, List[int]]]) -> List[np.ndarray]:
    min_position = np.asarray([p for _, p in data]).min(axis=0)
    pad_lower = [p - min_position for _, p in data]
    max_shape = np.asarray([(aim.shape + pl) for (aim, _), pl in zip(data, pad_lower)]).max(axis=0)
    pad_upper = [(max_shape - (aim.shape + pl)) for (aim, _), pl in zip(data, pad_lower)]
    return [
        np.pad(aim, tuple([(l, u) for l, u in zip(pl, pu)]), "constant")
        for (aim, _), pl, pu in zip(data, pad_lower, pad_upper)
    ]


def get_bounding_box(img):
    # code is taken from: https://stackoverflow.com/a/31402351

    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return rmin, rmax, cmin, cmax, zmin, zmax


def main():
    # args = create_parser().parse_args()
    args = Namespace()
    args.image_dir = "/Users/nathanneeteson/Documents/Data/Images/SALTAC/visit_1/tibia"

    image_fn_list = glob(os.path.join(args.image_dir, "*_?.AIM"))

    reader = vtkboneAIMReader()
    reader.DataOnCellsOff()

    for image_fn in image_fn_list:

        try:

            image, image_position = read_image_to_numpy(reader, image_fn, ImageType.DENSITY)
            cort, cort_position = read_image_to_numpy(reader, image_fn.replace(".AIM", "_CORT.AIM"), ImageType.MASK)
            trab, trab_position = read_image_to_numpy(reader, image_fn.replace(".AIM", "_TRAB.AIM"), ImageType.MASK)

            roi_codes_dict = get_roi_codes(image_fn[-5])
            for side, roi_code_list in roi_codes_dict.items():
                try:
                    os.mkdir(os.path.join(args.image_dir, side))
                except FileExistsError:
                    print(f"Directory already exists: {os.path.join(args.image_dir, side)}")
                aims = [(image, image_position), (cort, cort_position), (trab, trab_position)]
                for roi_code in roi_code_list:
                    aims.append(
                        read_image_to_numpy(
                            reader,
                            image_fn.replace(".AIM", f"_ROI{roi_code}_MASK.AIM"),
                            ImageType.MASK
                        )
                    )
                aims = align_aims(aims)  # 0-image, 1-cort, 2-trab, 3-roi1, 4-roi2, 5-roi3, 6-roi4
                x0, x1, y0, y1, z0, z1 = get_bounding_box(aims[3] | aims[4] | aims[5] | aims[6])
                cropped_image = aims[0][x0:x1, y0:y1, z0:z1]
                cropped_cort = aims[1][x0:x1, y0:y1, z0:z1]
                cropped_trab = aims[2][x0:x1, y0:y1, z0:z1]
                np.savez_compressed(
                    os.path.join(args.image_dir, side, f"{Path(image_fn).stem}.npz"),
                    image=cropped_image, cort_mask=cropped_cort, trab_mask=cropped_trab
                )
                print(f"succeeded for {image_fn}")

        except FileNotFoundError:
            print(f"failed for {image_fn}")








if __name__ == "__main__":
    main()
