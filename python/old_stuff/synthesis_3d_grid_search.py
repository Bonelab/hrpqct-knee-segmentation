import torch
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from blpytorchlightning.tasks.SegmentationTask import SegmentationTask
from blpytorchlightning.dataset_components.PickledDataset import PickledDataset
from blpytorchlightning.models.Synthesis3D import Synthesis3D


def create_parser():
    parser = ArgumentParser(
        description='HRpQCT Segmentation 2.5D UNet - 3D Synthesis FCN Grid Search Script',
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
        '--label', '-l', type=str, default='synthesis3d', metavar='STR',
        help='base title to use when saving logs and model checkpoints'
    )
    parser.add_argument(
        '--version', '-v', type=int, default=None, metavar='N',
        help='version number for logging'
    )
    parser.add_argument(
        '--num-workers', '-w', type=int, default=6, metavar='N',
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
        '--batch-size', '-bs', type=int, default=4, metavar='N',
        help='number of samples per minibatch'
    )
    parser.add_argument(
        '--num-gpus', '-ng', type=int, default=0, metavar='N',
        help='number of GPUs to use'
    )
    parser.add_argument(
        '--model-filters', '-mf', type=int, nargs='+',
        default=[8], metavar='N',
        help='number of filters in each layer of the 3D synthesis network'
    )
    parser.add_argument(
        '--channels-per-group', '-c', type=int, default=16, metavar='N',
        help='channels per group in GroupNorm'
    )
    parser.add_argument(
        '--dropout', '-d', type=float, default=0.1, metavar='D',
        help='dropout probability'
    )

    return parser


def train_model(hparams, training_dataloader, validation_dataloader):
    # reset the datasets
    training_dataloader.dataset.reset()
    validation_dataloader.dataset.reset()

    # create model
    model_kwargs = {
        'input_channels': 4,  # img, ax, sag, cor masks
        'output_classes': 1,  # binary classification problem for now
        'num_filters': hparams['model_filters'],
        'channels_per_group': hparams['channels_per_group'],
        'dropout': hparams['dropout']
    }
    model = Synthesis3D(**model_kwargs)
    model.float()

    # create task
    task = SegmentationTask(
        model,
        BCEWithLogitsLoss(),
        learning_rate=hparams['learning_rate']
    )

    # create logger
    csv_logger = CSVLogger(
        hparams['logs_dir'],
        name=hparams['label'],
        version=hparams['version']
    )
    csv_logger.log_hyperparams(hparams)

    # create callback
    early_stopping = EarlyStopping(
        monitor='val_dsc',
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

    try:
        train_model(hparams, training_dataloader, validation_dataloader)
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print('Ran out of memory, try again with batch size of 1')
            torch.cuda.empty_cache()
            hparams['batch_size'] = 1
            dataloader_kwargs = {
                'batch_size': hparams['batch_size'],
                'num_workers': hparams['num_workers'],
                'pin_memory': True
            }
            training_dataloader = DataLoader(training_dataset, shuffle=True, **dataloader_kwargs)
            validation_dataloader = DataLoader(validation_dataset, **dataloader_kwargs)
            train_model(hparams, training_dataloader, validation_dataloader)
        else:
            raise e


if __name__ == '__main__':
    main()
