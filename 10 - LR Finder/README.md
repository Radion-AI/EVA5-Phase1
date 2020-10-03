# Session 10 - LR Finder and Reduce LR on Plateau

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1eo19MODMBfDRy8m6YjIeZs6OkB1bdouh)

This assignment targets an accuracy of 88 percent on CIFAR10 dataset by implementing the following.

- Implement LRFinder to find best initial LR.
- Implement Reduce LR on Plateau.
- Cut out in Albumentations should be present.
- Use SGD with momentum

## Model Used
Resnet18

## Results

Final Accuracy = 90.14%<br>
Highest Accuracy = 90.14%<br>
Epochs used = 50<br>
Best Initial LR = 0.0074

### LR Finder Curve

![alt text](./images/lrfinder_bestlr.png)

### Train and Test curves

![alt text](./images/accuracy_change.png)


### Incorrect Predictions

![alt text](./images/incorrect_images.png)

### Gradcam on Incorrect Predictions
![alt text](./images/incorrect_cam_images.png)



## Group Members
- Vishwajeet Pratap Singh (vishwajeet.pratapsingh2207@gmail.com)
- Happy Singh (hsingh0805@gmail.com)
