import numpy as np
import os
import yaml
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from torch.nn import BCEWithLogitsLoss
from datasets.AIMLoader import AIMLoader
from datasets.PatchSampler import PatchSampler
from datasets.HRpQCTTransformer import HRpQCTTransformer
from datasets.HRpQCTDataset import HRpQCTDataset
from datasets.ComposedTransformers import ComposedTransformers
from datasets.UNet2DTransformer import UNet2DTransformer
from lightning_modules.SegmentationTask import SegmentationTask
from models.UNet2D import UNet2D


def create_parser():
    parser = ArgumentParser(
        description='HRpQCT Segmentation 2.5D UNet - 3D Fusion Preprocessing Script',
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
    parser.add_argument(
        '--unet2d-logs-dir', '-uld', type=str, default='./logs', metavar='STR',
        help='top-level directory for logging from 2D UNet training'
    )
    parser.add_argument(
        '--unet2d-label', '-ul', type=str, default='unet2d', metavar='STR',
        help='base title used when saving logs and model checkpoints from 2D UNet'
    )
    parser.add_argument(
        '--unet2d-version', '-uv', type=str, default='version_0', metavar='STR',
        help='version name for the best 2D UNet'
    )
    parser.add_argument(
        '--unet2d-ckpt-file', '-uc', type=str, default='*.ckpt', metavar='STR',
        help='specific filename of the final checkpoint of the 2D UNet'
    )

    return parser


def load_unet_2d(log_dir, label, version, ckpt_file):
    hparams_path = os.path.join(log_dir, label, version, 'hparams.yaml')
    ckpt_path = os.path.join(log_dir, label, version, 'checkpoints', ckpt_file)
    with open(hparams_path, 'r') as f:
        hparams = yaml.safe_load(f)
    model = UNet2D(
        1, 1,
        hparams['model_filters'],
        hparams['channels_per_group'],
        hparams['dropout']
    )
    model.float()
    task = SegmentationTask(
        model, BCEWithLogitsLoss(),
        learning_rate=hparams['learning_rate']
    )
    trained_task = task.load_from_checkpoint(
        ckpt_path,
        model=model,
        loss_function=BCEWithLogitsLoss(),
        learning_rate=hparams['learning_rate']
    )
    return trained_task.model


def main():
    # get parameters from command line
    args = create_parser().parse_args()

    # load the best 2D UNet
    unet_2d = load_unet_2d(
        args.unet2d_logs_dir,
        args.unet2d_label,
        args.unet2d_version,
        args.unet2d_ckpt_file
    )

    # create dataset
    file_loader = AIMLoader(args.data_dir, '*_*_??.AIM')
    sampler = PatchSampler(patch_width=args.patch_width)
    hrpqct_transformer = HRpQCTTransformer(intensity_bounds=[args.min_density, args.max_density])
    unet_2d_transformer = UNet2DTransformer(unet_2d)
    composed_transformers = ComposedTransformers([hrpqct_transformer, unet_2d_transformer])
    dataset = HRpQCTDataset(file_loader, sampler, composed_transformers)

    # pickle the samples
    if not args.idx_end:
        args.idx_end = len(dataset)
    idxs = np.arange(args.idx_start, args.idx_end, dtype=int)
    dataset.pickle_dataset(args.pickle_dir, idxs, args.epochs, args=args)


if __name__ == '__main__':
    main()
