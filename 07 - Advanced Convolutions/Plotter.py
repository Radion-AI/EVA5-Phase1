import os
import numpy as np
import matplotlib.pyplot as plt
import torch


def class_level_accuracy(model, loader, device, classes):
    """Print test accuracy for each class in dataset.

    Args:
        model: Model instance.
        loader: Data loader.
        device: Device where data will be loaded.
        classes: List of classes in the dataset.
    """

    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))

    with torch.no_grad():
        for _, (images, labels) in enumerate(loader, 0):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()

            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))


def plot_graph(values, metric):
    # Initialize a figure
    fig = plt.figure(figsize=(7, 5))

    # Plot values
    plt.plot(values)

    # Set plot title
    plt.title(f'Validation {metric}')

    # Label axes
    plt.xlabel('Epoch')
    plt.ylabel(metric)

    # Set legend
    location = 'upper' if metric == 'Loss' else 'lower'

    # Save plot
    fig.savefig(f'{metric.lower()}_change.png')


def save_and_show_result(data, classes):
    """Display 25 misclassified images.

    Args:
        data: Contains model predictions and labels.
    """

    # Create directories for saving data
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'predictions'
    )
    if not os.path.exists(path):
        os.makedirs(path)

    # Initialize plot
    row_count = -1
    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    fig.tight_layout()

    for idx, result in enumerate(data):

        # If 25 samples have been stored, break out of loop
        if idx > 24:
            break
        
        rgb_image = np.transpose(result['image'], (1, 2, 0)) / 2 + 0.5
        label = result['label'].item()
        prediction = result['prediction'].item()

        # Plot image
        if idx % 5 == 0:
            row_count += 1
        axs[row_count][idx % 5].axis('off')
        axs[row_count][idx % 5].set_title(f'Label: {classes[label]}\nPrediction: {classes[prediction]}')
        axs[row_count][idx % 5].imshow(rgb_image)
    
    # Save image
    fig.savefig(f'{path}/incorrect_predictions.png', bbox_inches='tight')
