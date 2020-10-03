import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_metric(data, metric, legend_loc='lower right'):

    single_plot = True
    if type(data) == dict:
        single_plot = False
    
    # Initialize a figure
    fig = plt.figure(figsize=(7, 5))

    # Plot data
    if single_plot:
        plt.plot(data)
    else:
        plots = []
        for value in data.values():
            plots.append(plt.plot(value)[0])

    # Set plot title
    plt.title(f'{metric} Change')

    # Label axes
    plt.xlabel('Epoch')
    plt.ylabel(metric)

    if not single_plot: # Set legend
        plt.legend(
            tuple(plots), tuple(data.keys()),
            loc=legend_loc,
            shadow=True,
            prop={'size': 15}
        )
      # Save plot
    fig.savefig(f'{metric.lower()}_change.png')


def plot_images(rows, columns, images, class_names, cls_true, metric, cls_pred=None, smooth=True):

    assert len(images) == len(cls_true) == rows * columns
    images = [np.array(np.transpose(np.array(image), (1, 2, 0)) / 2 + 0.5) for image in images]
    images = np.array(images)
    # Create figure with sub-plots.
    fig, axes = plt.subplots(rows, columns, figsize = (8, 8))

    # Adjust vertical spacing if we need to print ensemble and best-net.
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    # Interpolation type
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    for i, ax in enumerate(axes.flat):

        # Plot image.
        ax.imshow(images[i, :, :, :],
                  interpolation=interpolation)
            
        # Name of the true class.
        cls_true_name = class_names[cls_true[i]]

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true_name)
        else:
            # Name of the predicted class.
            cls_pred_name = class_names[cls_pred[i]]

            xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    fig.savefig(f'{metric.lower()}_images.png')
    plt.show()

def get_correct_samples(testloader, model, device, count):
    correct_images, true_labels = [], []
    with torch.no_grad():
        for _, (images, labels) in enumerate(testloader):
            img_batch = images  # This is done to keep data in CPU
            labels = labels
            images = images.to(device)  # Get samples
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.cpu().data.numpy()
            labels = labels.cpu().data.numpy()
            cnt = 0
            for i in range(len(labels)):
                label = labels[i]
                if predicted[i] == label:
                    cnt = cnt + 1
                    if cnt > count:
                        return correct_images, true_labels
                    correct_images.append(img_batch[i])
                    true_labels.append(label)
    return correct_images, true_labels

def get_incorrect_predictions(testloader, model, device, count):
    incorrect_images, incorrect_labels, true_labels = [], [], []
    with torch.no_grad():
        for _, (images, labels) in enumerate(testloader):
            img_batch = images  # This is done to keep data in CPU
            labels = labels
            images = images.to(device)  # Get samples
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.cpu().data.numpy()
            labels = labels.cpu().data.numpy()
            # cnt = 0
            for i in range(len(labels)):
                label = labels[i]
                if predicted[i] != label:
                    # print(predicted[i], label)
                    # cnt = cnt + 1
                    # print(cnt)
                    if len(true_labels) >= count:
                        return incorrect_images, incorrect_labels, true_labels
                    incorrect_images.append(img_batch[i])
                    incorrect_labels.append(predicted[i])
                    true_labels.append(label)
    return incorrect_images, incorrect_labels, true_labels

def plot_cam_view(rows, columns, images, class_names, cls_true, metric, cls_pred=None, smooth=True):

    assert len(images) == len(cls_true) == rows * columns
    fig, axes = plt.subplots(rows, columns, figsize = (8, 8))

    # Adjust vertical spacing if we need to print ensemble and best-net.
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    # Interpolation type
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    for i, ax in enumerate(axes.flat):

        # Plot image.
        ax.imshow(images[i],
                  interpolation=interpolation)
            
        # Name of the true class.
        cls_true_name = class_names[cls_true[i]]

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true_name)
        else:
            # Name of the predicted class.
            cls_pred_name = class_names[cls_pred[i]]

            xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    fig.savefig(f'{metric.lower()}_images.png')
    plt.show()