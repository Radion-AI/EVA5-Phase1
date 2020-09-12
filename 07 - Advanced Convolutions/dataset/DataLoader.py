from dataset.DataDownloader import download_cifar10_dataset
from dataset.DataPreProcessor import transform_data, data_loader


def dataset_classes():
  # Returns the classes of Cifar10 dataset
    return (
        'plane', 'car', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    )


def cifar10_dataset(batch_size, cuda_present, num_workers, train=True, augment=False, rotation=0.0):
    """Downloading and creating Cifar10 dataset.

    Args:
        batch_size: size of batch (Images in one batch).
        cuda_present: True is GPU is available.
        num_workers: How many subprocesses to use for data loading.
        train: If True, download training data else test data.
            Defaults to True.
        augment: Whether to apply data augmentation.
            Defaults to False.
        rotation: Angle of rotation of images for image augmentation.
            Defaults to 0. It won't be needed if augmentation is False.
    
    Returns:
        returns the instance of Dataloader.
    """

    # Define data transformations
    if train:
        transforms = transform_data(
            augment, rotation
        )
    else:
        transforms = transform_data()

    # Download training and validation dataset
    data = download_cifar10_dataset(train=train, transform=transforms)

    # create and return dataloader
    return data_loader(data, batch_size, num_workers, cuda_present)
