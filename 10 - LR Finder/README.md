# Session 9 - Data Augmentation

This assignment targets an accuracy of 88 percent on CIFAR10 dataset by implementing the following.

- Implement LRFinder to find best initial LR.
- Implement Reduce LR on Plateau.
- Cut out in Albumentations should be present.
- Implement

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
