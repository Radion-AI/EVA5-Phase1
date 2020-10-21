# Session 7 - Advanced Convolutions
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GlPmQ-LtFGaOPguOkn9ETcbspHhj0YC7)

This assignment aims to achieve 80% percent of accuracy on CIFAR10 with following conditions :

- Total Receptive Field must be more than 44
- One of the layers must use Depthwise Separable Convolution
- One of the layers must use Dilated Convolution
- Must use Global Average pooling (GAP)
- Add Fully Connected(FC) after GAP

## Model Architecture
![alt text](./images/S7_Architecture.png)

## Results

Accuracy = 84.43%<br>
Epochs used = 40<br>
Parameters = 94,218<br><br>

### Change in validation loss
![alt text](./images/TestLoss.png)


### Change in validation accuracy
![alt text](./images/TestAccuracy.png)


### Setup on Local System
```bash
pip3 install -r packages.txt
```

## Group Members
- Vishwajeet Pratap Singh (vishwajeet.pratapsingh2207@gmail.com)
- Happy Singh (hsingh0805@gmail.com)
