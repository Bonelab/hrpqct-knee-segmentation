import itertools
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from torch.nn import NLLLoss
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from lightning_modules.SegmentationEmbeddingTask import SegmentationEmbeddingTask
from datasets.PickledDataset import PickledDataset
from models.UNet2D import UNet2D
from utils.embedding_conversion import create_convert_embeddings_to_predictions
from loss_functions.CurvatureLoss import CurvatureLoss
from loss_functions.MagnitudeGradientSDTLoss import MagnitudeGradientSDTLoss


def create_parser():
    parser = ArgumentParser(
        description='HRpQCT Segmentation 2.5D UNet - 2D UNet Grid Search Script',
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
        '--logs-dir', '-ld', type=str, default='./logs', metavar='STR',
        help='top-level directory for logging'
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
        '--epochs', '-e', type=int, default=200, metavar='N',
        help='number of epochs to train for'
    )
    parser.add_argument(
        '--early-stopping-patience', '-esp', type=int, default=10, metavar='N',
        help='number of epochs to train for'
    )
    parser.add_argument(
        '--log-every-n-steps', '-lens', type=int, default=20, metavar='N',
        help='number of epochs to train for'
    )
    parser.add_argument(
        '--learning-rate', '-lr', type=float, default=0.001, metavar='LR',
        help='learning rate for the optimizer'
    )
    parser.add_argument(
        '--batch-size', '-bs', type=int, default=32, metavar='N',
        help='number of samples per minibatch'
    )
    parser.add_argument(
        '--curvature-threshold', type=float, default=0.005, metavar='D',
        help='maximum curvature above which excess curvature will be penalized, units: 1/um'
    )
    parser.add_argument(
        '--voxel-width', type=float, default=61, metavar='D',
        help='isotropic voxel width, units: um'
    )
    parser.add_argument(
        '--heaviside-epsilon', type=float, default=0.1, metavar='D',
        help='scaling parameter for the approximate heaviside function'
    )
    parser.add_argument(
        '--num_gpus', '-ng', type=int, default=0, metavar='N',
        help='number of GPUs to use'
    )

    return parser


def train_model(hparams, training_dataloader, validation_dataloader):
    # reset the datasets
    training_dataloader.dataset.reset()
    validation_dataloader.dataset.reset()

    # create model
    model_kwargs = {
        'input_channels': 1,  # just one for now
        'output_classes': 2,  # peri, endo embeddings
        'num_filters': hparams['model_filters'],
        'channels_per_group': hparams['channels_per_group'],
        'dropout': hparams['dropout']
    }
    model = UNet2D(**model_kwargs)
    model.float()

    # create task
    task = SegmentationEmbeddingTask(
        model,
        create_convert_embeddings_to_predictions(hparams['heaviside_epsilon']),
        NLLLoss(),
        CurvatureLoss(hparams['voxel_width'], hparams['curvature_threshold']),
        MagnitudeGradientSDTLoss(hparams['voxel_width']),
        learning_rate=hparams['learning_rate'],
        lambda_curvature=hparams['lambda_curvature'],
        lambda_maggrad=hparams['lambda_maggrad']
    )

    # create logger
    csv_logger = CSVLogger(
        hparams['logs_dir'],
        name=hparams['label']
    )
    csv_logger.log_hyperparams(hparams)

    # create callback
    early_stopping = EarlyStopping(
        monitor='val_dsc_0',
        mode='max',
        patience=hparams['early_stopping_patience']
    )

    # create trainer
    trainer = Trainer(
        gpus=hparams['num_gpus'],
        max_epochs=hparams['epochs'],
        log_every_n_steps=hparams['log_every_n_steps'],
        logger=csv_logger,
        callbacks=[early_stopping]
    )

    # train
    trainer.fit(task, training_dataloader, validation_dataloader)


def main():
    # get the fixed hyperparas as cl args
    hparams = vars(create_parser().parse_args())

    # define the variable hyperparams
    model_filters_list = [
        [32, 64, 128, 256],
        [32, 64, 128, 256, 512],
        [32, 64, 128, 256, 512, 1024],
        [64, 128, 256, 512],
        [64, 128, 256, 512, 1024],
        [64, 128, 256, 512, 1024, 2048]
    ]
    channels_per_group_list = [4, 8, 16]
    dropout_list = [0.1, 0.2, 0.3]
    lambda_curvature_list = [1e-5, 1e-6, 0]
    lambda_maggrad_list = [1e-5, 1e-6, 0]

    # create the datasets
    training_dataset = PickledDataset(hparams['training_data_dir'])
    validation_dataset = PickledDataset(hparams['validation_data_dir'])

    # create dataloaders
    dataloader_kwargs = {
        'batch_size': hparams['batch_size'],
        'num_workers': hparams['num_workers'],
        'pin_memory': True
    }
    training_dataloader = DataLoader(training_dataset, shuffle=True, **dataloader_kwargs)
    validation_dataloader = DataLoader(validation_dataset, **dataloader_kwargs)

    hparams_lists = [
        model_filters_list,
        channels_per_group_list,
        dropout_list,
        lambda_curvature_list,
        lambda_maggrad_list
    ]

    # grid search iteration training loop
    for var_hparams in itertools.product(*hparams_lists):

        hparams['model_filters'] = var_hparams[0]
        hparams['channels_per_group'] = var_hparams[1]
        hparams['dropout'] = var_hparams[2]
        hparams['lambda_curvature'] = var_hparams[3]
        hparams['lambda_maggrad'] = var_hparams[4]
        try:
            train_model(hparams, training_dataloader, validation_dataloader)
        except Exception as e:
            print('')
            print('Failed due to exception.')
            print(f'hparams: {hparams}')
            print(f'Exception: {e}')
            print('')


if __name__ == '__main__':
    main()
