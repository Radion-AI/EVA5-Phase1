import matplotlib.pyplot as plt

def plot_metric(values, metric):
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


def plot_images(rows, columns, images, class_names, cls_true, cls_pred=None, smooth=True):

    assert len(images) == len(cls_true) == rows * columns

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
    plt.show()

def get_correct_samples(testloader, count):
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
                    if cnt >= count + 1:
                        return correct_images, true_labels
                    correct_images.append(img_batch[i])
                    true_labels.append(label)
    return correct_images, true_labels

def get_incorrect_predictions(testloader):
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
                    if len(true_labels) >= 25:
                        return np.array(incorrect_images), incorrect_labels, true_labels
                    incorrect_images.append(np.transpose(np.array(img_batch[i]), (1, 2, 0)) / 2 + 0.5)
                    incorrect_labels.append(predicted[i])
                    true_labels.append(label)
    incorrect_images = np.array(incorrect_images)
    return incorrect_images, incorrect_labels, true_labels

