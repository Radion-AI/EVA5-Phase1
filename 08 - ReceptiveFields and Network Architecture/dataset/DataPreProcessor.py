import torch
from torchvision import transforms

def transform_data(augment=False, rotation=0.0):
    """Creating the data transformations
    
    Args:
        augment: Apply data augmentation or not
            Default value is set to False.
        rotation: Angle of rotation in Augmentation
            Default value is 0. If augment is false then doesn't matter
    
    Returns:
        Transform object containing defined data transformations.
    """

    transforms_list = [
        # It converts the data to torch.FloatTensor
        transforms.ToTensor(),

        # normalizing the data with mean and standard deviation to keep values in range [-1, 1]
        # since there are 3 channels for each image,
        # we have to specify mean and std for each channel
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    if augment:
        transforms_list = [
            # Rotate image by 6 degrees
            transforms.RandomRotation((-rotation, rotation), fill=(1,))
        ] + transforms_list
    
    return transforms.Compose(transforms_list)


def data_loader(data, batch_size, num_workers, cuda_present):
    """Create data loader

    Args:
        data: Downloaded dataset.
        batch_size: Number of images to considered in each batch.
        num_workers: How many subprocesses to use for data loading.
        cuda: True is GPU is available.
    
    Returns:
        returns DataLoader instance.
    """

    loader_arguments = {
        'shuffle': True,
        'batch_size': batch_size
    }

    # Check if GPU exists or not
    if cuda_present:
        loader_arguments['num_workers'] = num_workers
        loader_arguments['pin_memory'] = True
    
    return torch.utils.data.DataLoader(data, **loader_arguments)
