import torch
import torchvision

class Downloader:
    def __init__(self, download_dir, dataset_name):
        self.download_dir = download_dir
        self.dataset_name = dataset_name
        # self.apply_transform = apply_transform
        self.downloader = {'CIFAR10' : torchvision.datasets.CIFAR10,
                            'CIFAR100' : torchvision.datasets.CIFAR100,
                            'MNSIT' : torchvision.datasets.MNIST}[dataset_name]

    def download_sample(self, train = True):
        return self.downloader(self.download_dir, download = True, train = train, transform = None)

    def download(self, transformations, train = True):
        return self.downloader(self.download_dir, download = True, train = train, transform = transformations)

