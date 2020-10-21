import torch.nn.functional as F
from tqdm import tqdm

from models.Regularizer import l1_regularization


def trainModel(model, loader, device, optimizer, loss_function, l1_factor = 0.0):
    """
    input parameters:
    instance of model, data loader, device, optimizer, loss function, L1 regularisation factor
    """

    model.train()
    pbar = tqdm(loader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar, 0):
        # Get samples
        data, target = data.to(device), target.to(device)

        # Set gradients to zero before starting backpropagation
        optimizer.zero_grad()

        # Predicting the output
        y_pred = model(data)

        # Calculating loss
        loss = l1_regularization(model, loss_function(y_pred, target), l1_factor)

        # Performing backpropagation
        loss.backward()
        optimizer.step()

        # Updating Progress Bar
        prediction = y_pred.argmax(dim=1, keepdim=True)
        correct += prediction.eq(target.view_as(prediction)).sum().item()
        processed += len(data)
        pbar.set_description(
            desc=f'Loss={loss.item():0.2f} Batch_ID={batch_idx} Accuracy={(100 * correct / processed):.2f}'
        )
