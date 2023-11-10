import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from blpytorchlightning.dataset_components.file_loaders.AIMLoader import AIMLoader
from blpytorchlightning.dataset_components.samplers.ForegroundPatchSampler import ForegroundPatchSampler
from blpytorchlightning.dataset_components.transformers.Rescaler import Rescaler
from blpytorchlightning.dataset_components.transformers.TensorConverter import TensorConverter
from blpytorchlightning.dataset_components.transformers.ComposedTransformers import ComposedTransformers
from blpytorchlightning.dataset_components.datasets.ComposedDataset import ComposedDataset


def create_parser():
    parser = ArgumentParser(
        description='HRpQCT Segmentation 3D Preprocessing Script',
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
        '--patch-width', '-pw', type=int, default=128, metavar='N',
        help='width of slice patch to use in training'
    )
    parser.add_argument(
        '--foreground-channel', '-fc', type=int, default=0, metavar='N',
        help='channel to use for centering patches'
    )
    parser.add_argument(
        '--probability', '-p', type=float, default=0.5, metavar='P',
        help='probability of sampling a foreground patch'
    )

    return parser


def main():
    # get parameters from command line
    args = create_parser().parse_args()

    # create dataset
    file_loader = AIMLoader(args.data_dir, '*_*_??.AIM')
    sampler = ForegroundPatchSampler(
        patch_width=args.patch_width,
        foreground_channel=args.foreground_channel,
        prob=args.probability
    )
    transformer = ComposedTransformers([
        Rescaler(intensity_bounds=[args.min_density, args.max_density]),
        TensorConverter()
    ])
    dataset = ComposedDataset(file_loader, sampler, transformer)

    # pickle the samples
    if not args.idx_end:
        args.idx_end = len(dataset)
    idxs = np.arange(args.idx_start, args.idx_end, dtype=int)
    dataset.pickle_dataset(args.pickle_dir, idxs, args.epochs, args=args)


if __name__ == '__main__':
    main()
