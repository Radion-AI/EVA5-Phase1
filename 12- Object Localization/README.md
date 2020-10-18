# Session 12 - Object Localisation

## Assignent-A

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FNJ4VzjgnzbW2NROHFaAav6qgefMOe0c)

This assignment targets an accuracy of more than 50 percent on TinyImageNet Dataset

## Model Used
Resnet18

## Results

Final Accuracy = 53.00%<br>
Highest Accuracy = 53.37%<br>
Epochs used = 50<br>
Best Initial LR = 0.0014

### LR Finder Curve

![alt text](./images/tinyimage_lrfinder.png)

### Train and Test curves

![alt text](./images/tinyimagenet_accuracy_change.png)


## Assignent-B

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Z2cUNqAQ4o_OaMO_DGtFZfizfDKObIsg#scrollTo=YTK--Qs2f3tN)

The goal is to download 50 images each of people wearing hardhat, vest, mask and boots and to annotate them. These images are exported in COCO JSON format after annotation. After that anchor boxes are found using K-Means Clustering.

### Number of clusters vs Mean IoU

![alt text](./images/kmeans_iou.png)


After running the algorithm on the dataset, The best k value was found at 4 and 7.

| Number of Clusters (k) | Mean IoU |                  Cluster Plot                  |                 Anchor Boxes                 |
| :--------------------: | :------: | :--------------------------------------------: | :------------------------------------------: |
|           4           |   0.54   | ![K4_cluster_plot](images/K4_cluster_plot.png) | ![K4_anchor_box](images/K4_anchor_box.png) |
|           7            |   0.54   | ![K7_cluster_plot](images/K7_cluster_plot.png) | ![K7_anchor_box](images/K7_anchor_box.png) |


## Group Members
- Vishwajeet Pratap Singh (vishwajeet.pratapsingh2207@gmail.com)
- Happy Singh (hsingh0805@gmail.com)
