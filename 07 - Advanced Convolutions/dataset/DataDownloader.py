import os

from torchvision import datasets


def download_cifar10_dataset(train=True, transform=None):
    """Downloading CIFAR10 dataset

    Args:
        train: If True, download training data else test data.
            Default value is set to True.
        transform: Data transformations to be applied on the data.
            Default value is toNone.
    
    Returns:
        It returns the downloaded dataset.
    """

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cifar10')
    return datasets.CIFAR10(
        data_path, train=train, download=True, transform=transform
    )
