# Session 6 - Regularization
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Pjomp2kf8IKm_4reOa5KUWCvc5GgCPuV)

This assignment aims to run the model for below versions for 25 epochs and then show the validation accuracy curves and loss change curves for all these versions :

- with L1 + BN
- with L2 + BN
- with L1 and L2 with BN
- with GBN
- with L1 and L2 with GBN

After that display 25 misclassified images for "with GBN" model


## Model Architecture
![alt text](./images/S6-Architecture.png)

## Results

### Change in validation loss
![alt text](./images/ValidationLoss.png)


### Change in validation accuracy
![alt text](./images/ValidationAccuracy.png)


## Misclassified Images
![alt text](./images/IncorrectImages.png)

### With GBN (Ghost batch Normalization)

### Setup on Local System
```bash
pip3 install -r packages.txt
```

## Group Members
- Vishwajeet Pratap Singh (vishwajeet.pratapsingh2207@gmail.com)
- Happy Singh (hsingh0805@gmail.com)