import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from blpytorchlightning.dataset_components.AIMLoader import AIMLoader
from blpytorchlightning.dataset_components.SlicePatchSampler import SlicePatchSampler
from blpytorchlightning.dataset_components.HRpQCTTransformer import HRpQCTTransformer
from blpytorchlightning.dataset_components.HRpQCTDataset import HRpQCTDataset


def create_parser():
    parser = ArgumentParser(
        description='HRpQCT Segmentation 2.5D UNet - 2D UNet Preprocessing Script',
        formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'data_dir', type=str, metavar='STR',
        help='main directory of the raw dataset'
    )
    parser.add_argument(
        'pickle_dir', type=str, metavar='STR',
        help='main directory to save the pickled dataset to'
    )
    parser.add_argument(
        '--idx-start', '-is', type=int, default=0, metavar='N',
        help='index to start pickling at'
    )
    parser.add_argument(
        '--idx-end', '-ie', type=int, default=None, metavar='N',
        help='index to end pickling at'
    )
    parser.add_argument(
        '--min-density', '-mind', type=float, default=-400, metavar='D',
        help='minimum physiologically relevant density in the image [mg HA/ccm]'
    )
    parser.add_argument(
        '--max-density', '-maxd', type=float, default=1400, metavar='D',
        help='maximum physiologically relevant density in the image [mg HA/ccm]'
    )
    parser.add_argument(
        '--epochs', '-e', type=int, default=200, metavar='N',
        help='number of epochs to train for'
    )
    parser.add_argument(
        '--patch-width', '-pw', type=int, default=160, metavar='N',
        help='width of slice patch to use in training'
    )

    return parser


def main():
    # get parameters from command line
    args = create_parser().parse_args()

    # create dataset
    file_loader = AIMLoader(args.data_dir, '*_*_??.AIM')
    sampler = SlicePatchSampler(patch_width=args.patch_width)
    transformer = HRpQCTTransformer(intensity_bounds=[args.min_density, args.max_density])
    dataset = HRpQCTDataset(file_loader, sampler, transformer)

    # pickle the samples
    if not args.idx_end:
        args.idx_end = len(dataset)
    idxs = np.arange(args.idx_start, args.idx_end, dtype=int)
    dataset.pickle_dataset(args.pickle_dir, idxs, args.epochs, args=args)


if __name__ == '__main__':
    main()
