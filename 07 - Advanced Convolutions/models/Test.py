import torch
import torch.nn.functional as F


def testModel(model, loader, device, criterion, losses, accuracies, incorrect_samples):

    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            img_batch = data  # This is done to keep data in CPU
            data, target = data.to(device), target.to(device)  # Get samples
            output = model(data)  # Get trained model output
            val_loss += criterion(output, target).item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            result = pred.eq(target.view_as(pred))

            # Save incorrect samples
            if len(incorrect_samples) < 25:
                for i in range(loader.batch_size):
                    if not list(result)[i]:
                        incorrect_samples.append({
                            'prediction': list(pred)[i],
                            'label': list(target.view_as(pred))[i],
                            'image': img_batch[i]
                        })

            correct += result.sum().item()

    val_loss /= len(loader.dataset)
    losses.append(val_loss)
    accuracies.append(100. * correct / len(loader.dataset))

    print(
        f'\nValidation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(loader.dataset)} ({accuracies[-1]:.2f}%)\n'
    )
