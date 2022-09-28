import matplotlib.pyplot as plt

from blpytorchlightning.dataset_components.AIMLoader import AIMLoader
from blpytorchlightning.dataset_components.AIMLoaderRAM import AIMLoaderRAM
from blpytorchlightning.dataset_components.PatchSampler import PatchSampler
from blpytorchlightning.dataset_components.SlicePatchSampler import SlicePatchSampler
from blpytorchlightning.dataset_components.HRpQCTTransformer import HRpQCTTransformer
from blpytorchlightning.dataset_components.HRpQCTDataset import HRpQCTDataset


def main():
    data_dir = '/Users/nathanneeteson/Documents/Data/test'
    data_pattern = '*_*_??.AIM'

    loader = AIMLoaderRAM(data_dir, data_pattern)
    sampler = SlicePatchSampler()
    transformer = HRpQCTTransformer()

    dataset = HRpQCTDataset(loader, sampler, transformer)

    image, masks = dataset[0]

    print(image.shape)
    print(masks.shape)

    colormaps = ['Blues', 'Greens', 'Reds']

    plt.figure(figsize=(8, 8))

    plt.imshow(image[0, ...], cmap='Greys_r')

    for i in range(3):
        plt.imshow(masks == i, cmap=colormaps[i], alpha=0.5 * (masks == i))

    plt.show()


if __name__ == '__main__':
    main()
