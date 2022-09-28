from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from lightning_modules.SegmentationTask import SegmentationTask
from blpytorchlightning.dataset_components.AIMLoader import AIMLoader
from blpytorchlightning.dataset_components.AIMLoaderRAM import AIMLoaderRAM
from blpytorchlightning.dataset_components.SlicePatchSampler import SlicePatchSampler
from blpytorchlightning.dataset_components.HRpQCTTransformer import HRpQCTTransformer
from blpytorchlightning.dataset_components.HRpQCTDataset import HRpQCTDataset
from blpytorchlightning.models.UNet2D import UNet2D


def create_parser():
    parser = ArgumentParser(
        description='HRpQCT Segmentation 2.5D UNet - 2D UNet Training Script',
        formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--training-data-dir', '-td', type=str, default='./data/train', metavar='STR',
        help='main directory of the training dataset'
    )
    parser.add_argument(
        '--validation-data-dir', '-vd', type=str, default='./data/validate', metavar='STR',
        help='main directory of the validation dataset'
    )
    parser.add_argument(
        '--label', '-l', type=str, default='unet2d', metavar='STR',
        help='base title to use when saving logs and model checkpoints'
    )
    parser.add_argument(
        '--num-workers', '-w', type=int, default=8, metavar='N',
        help='number of CPU workers to use to load data in parallel'
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
        '--model-filters', '-f', type=int, nargs='+', default=[64, 128, 256, 512, 1024], metavar='N',
        help='sequence of filters in U-Net layers'
    )
    parser.add_argument(
        '--channels-per-group', '-c', type=int, default=16, metavar='N',
        help='channels per group in GroupNorm'
    )
    parser.add_argument(
        '--dropout', '-d', type=float, default=0.1, metavar='D',
        help='dropout probability'
    )
    parser.add_argument(
        '--epochs', '-e', type=int, default=200, metavar='N',
        help='number of epochs to train for'
    )
    parser.add_argument(
        '--learning-rate', '-lr', type=float, default=0.001, metavar='LR',
        help='learning rate for the optimizer'
    )
    parser.add_argument(
        '--patch-width', '-pw', type=int, default=160, metavar='N',
        help='width of slice patch to use in training'
    )
    parser.add_argument(
        '--batch-size', '-bs', type=int, default=32, metavar='N',
        help='number of samples per minibatch'
    )
    parser.add_argument(
        '--num_gpus', '-ng', type=int, default=0, metavar='N',
        help='number of GPUs to use'
    )
    parser.add_argument(
        '--version', '-v', type=int, default=None, metavar='N',
        help='version number for logging'
    )
    parser.add_argument(
        '--dataset-in-memory', '-dm', action='store_true', default=False,
        help='load the whole dataset into memory for faster access'
    )

    return parser


def main():
    # get parameters from command line
    args = create_parser().parse_args()

    # create the model
    model_kwargs = {
        'input_channels': 1,  # just one for now
        'output_classes': 3,  # cort, trab, back
        'num_filters': args.model_filters,
        'channels_per_group': args.channels_per_group,
        'dropout': args.dropout
    }
    model = UNet2D(**model_kwargs)
    model.float()

    # create the task
    task = SegmentationTask(
        model, CrossEntropyLoss(),
        learning_rate=args.learning_rate
    )

    # create datasets
    if args.dataset_in_memory:
        FileLoader = AIMLoaderRAM
    else:
        FileLoader = AIMLoader
    training_file_loader = FileLoader(args.training_data_dir, '*_*_??.AIM')
    validation_file_loader = FileLoader(args.validation_data_dir, '*_*_??.AIM')
    sampler = SlicePatchSampler(patch_width=args.patch_width)
    transformer = HRpQCTTransformer(intensity_bounds=[args.min_density, args.max_density])

    training_dataset = HRpQCTDataset(training_file_loader, sampler, transformer)
    validation_dataset = HRpQCTDataset(validation_file_loader, sampler, transformer)

    # create dataloaders
    dataloader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'pin_memory': True
    }
    training_dataloader = DataLoader(training_dataset, shuffle=True, **dataloader_kwargs)
    validation_dataloader = DataLoader(validation_dataset, **dataloader_kwargs)

    # create loggers
    csv_logger = CSVLogger('./logs', name=args.label, version=args.version)
    csv_logger.log_hyperparams(args)

    # create a Trainer
    trainer = Trainer(
        gpus=args.num_gpus,
        max_epochs=args.epochs,
        log_every_n_steps=20,
        logger=csv_logger
    )
    trainer.fit(task, training_dataloader, validation_dataloader)


if __name__ == '__main__':
    main()
