import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import yaml
import os

from torch.nn import BCEWithLogitsLoss, L1Loss
from blpytorchlightning.tasks.SegmentationTask import SegmentationTask
from blpytorchlightning.tasks.SynthesisTask import SynthesisTask
from blpytorchlightning.tasks.SeGANTask import SeGANTask

from datasets.AIMLoader import AIMLoader
from datasets.AIMLoaderRAM import AIMLoaderRAM
from datasets.SlicePatchSampler import SlicePatchSampler
from datasets.PatchSampler import PatchSampler
from datasets.HRpQCTTransformer import HRpQCTTransformer
from datasets.HRpQCTDataset import HRpQCTDataset

from datasets.PickledDataset import PickledDataset
from datasets.UNet2DTransformer import UNet2DTransformer

from models.UNet2D import UNet2D
from models.Synthesis3D import Synthesis3D
from models.SeGAN import get_segmentor_and_discriminators

from utils.error_metrics import dice_similarity_coefficient


def main_synthesis():
    data_dir = '/Users/nathanneeteson/Documents/Data/test_pickle_3d'

    unet2d = UNet2D(1, 1, [8, 16, 32], 4, 0.1)
    synthesis3d = Synthesis3D(4, 1, [8, 4], 4, 0.1)

    unet2d.eval()
    synthesis3d.eval()

    segmentation_task = SegmentationTask(unet2d, BCEWithLogitsLoss(), 0.001)
    synthesis_task = SynthesisTask(synthesis3d, segmentation_task, BCEWithLogitsLoss(), 0.001)

    dataset = PickledDataset(data_dir)

    image, mask = dataset[0]
    image = image.unsqueeze(0)  # add batch dimension like the dataloader will do
    mask = mask.unsqueeze(0)

    start = time.time()
    image = synthesis_task._frozen_segmentation(image)
    print(f'frozen segmentation took {time.time() - start:0.3f} seconds')

    '''

    start = time.time()
    loss = synthesis_task.training_step((image,mask),0)
    print(f'computing loss for one step took {time.time()-start:0.3f} seconds')



    print(loss)

    with torch.no_grad():
        pred = synthesis3d(image)

    print(f'image shape: {image.shape}')
    print(f'pred shape: {pred.shape}')
    print(f'mask shape: {mask.shape}')

    fig, axs = plt.subplots(1,6)

    for i in range(4):
        axs[i].imshow(image[0,i,80,:,:])

    axs[4].imshow(pred[0,0,80,:,:])

    axs[5].imshow(mask[0,0,80,:,:])

    plt.show()

    '''


def main_3d():
    data_dir = '/Users/nathanneeteson/Documents/Data/test_pickle_3d'

    '''
    file_loader = AIMLoaderRAM(data_dir,'*_*_??.AIM')
    sampler = PatchSampler(patch_width=128)
    transformer = HRpQCTTransformer()
    dataset = HRpQCTDataset(file_loader,sampler,transformer)
    '''

    dataset = PickledDataset(data_dir)

    image, mask = dataset[0]
    image = image.unsqueeze(0)  # add batch dimension like the dataloader will do
    mask = mask.unsqueeze(0)

    fig, axs = plt.subplots(1, 5)

    for i in range(4):
        axs[i].imshow(image[0, i, 80, :, :])

    axs[4].imshow(mask[0, 0, 80, :, :])

    plt.show()


def main_pickle():
    data_dir = '/Users/nathanneeteson/Documents/Data/test'
    pickle_dir = '/Users/nathanneeteson/Documents/Data/test_pickle'

    file_loader = AIMLoader(data_dir, '*_*_??.AIM')
    sampler = SlicePatchSampler(patch_width=160)
    transformer = HRpQCTTransformer()
    dataset = HRpQCTDataset(file_loader, sampler, transformer)

    dataset.pickle_dataset(pickle_dir, 10)


def main_pickle_load():
    pickle_dir = './data/train'

    dataset = PickledDataset(pickle_dir)

    num_samples = 10

    fig, axs = plt.subplots(num_samples, 3)

    for i in range(num_samples):

        print(i)

        image, mask = dataset[i % len(dataset)]

        axs[i][0].imshow(image[0, :, :], cmap='Greys_r', clim=[-1, 1])
        axs[i][1].imshow(mask, cmap='Greys_r', clim=[0, 2])

        for j in range(2):
            axs[i][j].set_axis_off()

    plt.subplots_adjust(
        left=0.01,
        right=0.99,
        bottom=0.01,
        top=0.99,
        wspace=0.01,
        hspace=0.01
    )
    plt.show()


def main_old():
    data_dir = '/Users/nathanneeteson/Documents/Data/test'

    file_loader = AIMLoaderRAM(data_dir, '*_*_??.AIM')
    sampler = SlicePatchSampler(patch_width=160)
    transformer = HRpQCTTransformer()
    dataset = HRpQCTDataset(file_loader, sampler, transformer)

    # dataset = RAMSliceDataset(data_dir,patch_width=160)

    num_samples = 5

    '''
    start = time.time()
    for _ in range(1000):
        sample = dataset[0]
    print(f'elapsed time: {time.time()-start}')
    '''

    fig, axs = plt.subplots(2, num_samples)

    for i in range(num_samples):

        print(i)

        image, mask = dataset[0]

        axs[0][i].imshow(image[0, :, :], cmap='Greys_r', clim=[-1, 1])
        axs[1][i].imshow(mask[0, :, :], cmap='Greys_r', clim=[0, 1])

        for j in range(2):
            axs[j][i].set_axis_off()

    plt.subplots_adjust(
        left=0.01,
        right=0.99,
        bottom=0.01,
        top=0.99,
        wspace=0.01,
        hspace=0.01
    )
    plt.show()


def main():
    LOG_DIR = 'logs'
    MODEL_DIR = 'unet2d'
    EXP_DIR = 'version_13549310'
    CKPT_DIR = 'checkpoints'
    HPARAMS_FILE = 'hparams.yaml'
    CKPT_FILE = 'epoch=49-step=1449.ckpt'

    HPARAMS_PATH = os.path.join('..', LOG_DIR, MODEL_DIR, EXP_DIR, HPARAMS_FILE)
    CKPT_PATH = os.path.join('..', LOG_DIR, MODEL_DIR, EXP_DIR, CKPT_DIR, CKPT_FILE)

    with open(HPARAMS_PATH, 'r') as f:
        hparams = yaml.safe_load(f)

    model = UNet2D(
        1, 1,
        num_filters=hparams['model_filters'],
        channels_per_group=hparams['channels_per_group'],
        dropout=hparams['dropout']
    )
    model.float()

    task = SegmentationTask(
        model, BCEWithLogitsLoss(),
        hparams['learning_rate']
    )

    trained_task = task.load_from_checkpoint(
        CKPT_PATH,
        model=model,
        loss_function=BCEWithLogitsLoss(),
        opt_kwargs={'lr': hparams['learning_rate']}
    )

    trained_task.eval()

    data_dir = '/Users/nathanneeteson/Documents/Data/test_hdf5'
    dataset = HDF5SliceDataset(data_dir, patch_width=160)

    z = 100
    image, mask = niftloader[0]
    pred = np.zeros_like(mask)

    image = torch.from_numpy(image).unsqueeze(0)

    img_slice = torch.from_numpy(image[..., z]).unsqueeze(0)


    num_samples = 5

    fig, axs = plt.subplots(3, num_samples)

    for i in range(num_samples):

        print(i)

        image, mask = dataset[0]

        with torch.no_grad():
            pred = torch.sigmoid(task.model(image.unsqueeze(0))[0, ...])

        axs[0][i].imshow(image[0, :, :], cmap='Greys_r', clim=[-1, 1])
        axs[1][i].imshow(mask[0, :, :], cmap='Greys_r', clim=[0, 1])
        axs[2][i].imshow(pred[0, :, :], cmap='Greys_r', clim=[0, 1])

        for j in range(3):
            axs[j][i].set_axis_off()

    plt.subplots_adjust(
        left=0.01,
        right=0.99,
        bottom=0.01,
        top=0.99,
        wspace=0.01,
        hspace=0.01
    )
    plt.show()


def main_check_segan():
    LOG_DIR = 'arc_logs'
    MODEL_DIR = 'SeGAN2D_Testing'
    EXP_DIR = 'version_2'
    CKPT_DIR = 'checkpoints'
    HPARAMS_FILE = 'hparams.yaml'
    CKPT_FILE = 'epoch=30-step=898.ckpt'

    HPARAMS_PATH = os.path.join('..', LOG_DIR, MODEL_DIR, EXP_DIR, HPARAMS_FILE)
    CKPT_PATH = os.path.join('..', LOG_DIR, MODEL_DIR, EXP_DIR, CKPT_DIR, CKPT_FILE)

    with open(HPARAMS_PATH, 'r') as f:
        hparams = yaml.safe_load(f)

    model_kwargs = {
        'input_channels': 1,  # just the densities
        'output_classes': 3,  # cort, trab, back
        'num_filters': hparams["model_filters"],
        'channels_per_group': hparams["channels_per_group"],
        'upsample_mode': hparams["upsample_mode"]
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
        learning_rate=hparams["learning_rate"]
    )

    trained_task = task.load_from_checkpoint(
        CKPT_PATH,
        segmentor=segmentor, discriminators=discriminators,
        loss_function=L1Loss(),
        learning_rate=hparams["learning_rate"]
    )

    trained_task.eval()

    data_dir = './data/train'
    dataset = PickledDataset(data_dir)

    num_samples = 5

    fig, axs = plt.subplots(4, num_samples)

    for i in range(num_samples):

        print(i)

        image, mask = dataset[0]

        with torch.no_grad():
            pred = torch.sigmoid(task(image.unsqueeze(0))[0, ...])

        pred = torch.argmax(pred, dim=0)

        axs[0][i].imshow(image[0, :, :], cmap='Greys_r', clim=[-1, 1])
        axs[1][i].imshow(mask, cmap='Greys_r', clim=[0, 2])
        axs[2][i].imshow(pred, cmap='Greys_r', clim=[0, 2])
        axs[3][i].imshow(
            pred != mask,
            cmap='Reds',
            clim=[0, 1],
            alpha=1
        )

        for j in range(4):
            axs[j][i].set_axis_off()

    plt.subplots_adjust(
        left=0.01,
        right=0.99,
        bottom=0.01,
        top=0.99,
        wspace=0.01,
        hspace=0.01
    )
    plt.show()


if __name__ == '__main__':
    main()
