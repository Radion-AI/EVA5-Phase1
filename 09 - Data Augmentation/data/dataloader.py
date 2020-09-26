import torch

class Dataloader:

    def getloader(dataset, batch_size, num_workers, train = True):
        return torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = train, num_workers = num_workers)
