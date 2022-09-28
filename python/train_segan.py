from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from torch.nn import L1Loss
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from blpytorchlightning.tasks.SeGANTask import SeGANTask
from blpytorchlightning.dataset_components.PickledDataset import PickledDataset
from blpytorchlightning.models.SeGAN import get_segmentor_and_discriminators


def create_parser():
    parser = ArgumentParser(
        description='HRpQCT Segmentation 2D SeGAN Training Script',
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
        '--label', '-l', type=str, default='SeGAN2D', metavar='STR',
        help='base title to use when saving logs and model checkpoints'
    )
    parser.add_argument(
        '--num-workers', '-w', type=int, default=8, metavar='N',
        help='number of CPU workers to use to load data in parallel'
    )
    parser.add_argument(
        '--model-filters', '-f', type=int, nargs='+', default=[64, 128, 256, 512], metavar='N',
        help='sequence of filters in SeGAN'
    )
    parser.add_argument(
        '--channels-per-group', '-c', type=int, default=16, metavar='N',
        help='channels per group in GroupNorm'
    )
    parser.add_argument(
        '--upsample-mode', '-us', type=str, default="bilinear", metavar='STR',
        help='method for upsampling in decoder'
    )
    parser.add_argument(
        '--is-3d', '-3d', action="store_true", default=False,
        help="set this flag to train a SeGAN with 3D convolutions for segmenting 3D image data"
    )
    parser.add_argument(
        '--epochs', '-e', type=int, default=200, metavar='N',
        help='number of epochs to train for'
    )
    parser.add_argument(
        '--early-stopping-patience', '-esp', type=int, default=100, metavar='N',
        help='number of epochs to train for'
    )
    parser.add_argument(
        '--log-every-n-steps', '-lens', type=int, default=20, metavar='N',
        help='number of epochs to train for'
    )
    parser.add_argument(
        '--learning-rate', '-lr', type=float, default=0.00002, metavar='LR',
        help='learning rate for the optimizer'
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

    return parser


def main():
    # get parameters from command line
    args = create_parser().parse_args()

    # create the model
    model_kwargs = {
        'input_channels': 1,  # just the densities
        'output_classes': 3,  # cort, trab, back
        'num_filters': args.model_filters,
        'channels_per_group': args.channels_per_group,
        'upsample_mode': args.upsample_mode,
        'is_3d': args.is_3d
    }
    segmentor, discriminators = get_segmentor_and_discriminators(**model_kwargs)
    segmentor.float()
    for d in discriminators:
        d.float()

    # create the task
    task = SeGANTask(
        segmentor,
        discriminators,
        L1Loss(),
        learning_rate=args.learning_rate
    )

    # create datasets
    training_dataset = PickledDataset(args.training_data_dir)
    validation_dataset = PickledDataset(args.validation_data_dir)

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

    # create callbacks
    early_stopping = EarlyStopping(
        monitor='val_dsc_0',
        mode='max',
        patience=args.early_stopping_patience
    )

    # create a Trainer
    trainer = Trainer(
        gpus=args.num_gpus,
        max_epochs=args.epochs,
        log_every_n_steps=args.log_every_n_steps,
        logger=csv_logger,
        callbacks=[early_stopping]
    )
    trainer.fit(task, training_dataloader, validation_dataloader)


if __name__ == "__main__":
    main()
