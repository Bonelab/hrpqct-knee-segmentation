import matplotlib.pyplot as plt

from datasets.AIMLoader import AIMLoader
from datasets.AIMLoaderRAM import AIMLoaderRAM
from datasets.PatchSampler import PatchSampler
from datasets.SlicePatchSampler import SlicePatchSampler
from datasets.HRpQCTTransformer import HRpQCTTransformer
from datasets.HRpQCTDataset import HRpQCTDataset

def main():
    data_dir = '/Users/nathanneeteson/Documents/Data/test'
    data_pattern = '*_*_??.AIM'

    loader = AIMLoaderRAM(data_dir,data_pattern)
    sampler = SlicePatchSampler()
    transformer = HRpQCTTransformer()

    dataset = HRpQCTDataset(loader, sampler, transformer)

    image, masks = dataset[0]

    print(image.shape)
    print(masks.shape)

    colormaps = ['Blues', 'Greens', 'Reds']

    plt.figure(figsize=(8,8))

    plt.imshow(image[0,...],cmap='Greys_r')

    for i in range(3):
        plt.imshow(masks[i,...],cmap=colormaps[i],alpha=(0.5)*masks[i,...])

    plt.show()

if __name__ == '__main__':
    main()
