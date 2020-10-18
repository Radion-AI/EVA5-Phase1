import torch

def val(model, loader, device, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)  # Get samples
            output = model(data)  # Get trained model output
            val_loss += criterion(output, target).item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            result = pred.eq(target.view_as(pred))

            correct += result.sum().item()

    val_loss /= len(loader.dataset)
    accuracy = 100. * correct / len(loader.dataset)

    print(
        f'\nValidation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(loader.dataset)} ({accuracy:.2f}%)\n'
    )
    return val_loss, accuracy