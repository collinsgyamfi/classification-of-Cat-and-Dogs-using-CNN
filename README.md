# Dog and Cat Image Classification using CNN

## Overview
This project aims to classify images of cats and dogs using a Convolutional Neural Network (CNN). The dataset is sourced from Kaggle and includes a balanced set of images for both classes. The project involves data preprocessing, exploratory data analysis, model training, and evaluation.

## Dataset
The dataset used in this project is sourced from Kaggle. It contains images of cats and dogs. After downloading the dataset, it should be unzipped and organized appropriately for processing. Ensure that the images are placed in separate folders for cats and dogs to facilitate easier loading and labeling during training.

## Data Preprocessing
Data preprocessing is essential to prepare the dataset for training the CNN model. The following steps outline the preprocessing workflow:

1. **Import Libraries**: Import necessary libraries for data handling and model building. Commonly used libraries include:
   - `numpy`
   - `pandas`
   - `matplotlib`
   - `seaborn`
   - `tensorflow` (which includes Keras)

   Example:
   ```python
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   import seaborn as sns
   from tensorflow.keras.preprocessing.image import ImageDataGenerator

- `tensorflow` (which includes Keras)

You can install the required libraries using pip:

```bash
pip install numpy pandas matplotlib seaborn tensorflow

## Create data frame
 Construct a DataFrame for input and output with labels. This DataFrame will map the image file paths to their corresponding labels (cat or dog).

## Data Augmentation
Use ImageDataGenerator from Keras for data augmentation. This technique enhances the model's robustness by generating new training images through transformations such as rotation, zoom, and horizontal flipping.

## Exploratory Data Analysis
Exploratory Data Analysis (EDA) is performed to understand the dataset better and visualize the distribution of classes.

Visualizing Images: Use Matplotlib to visualize a selection of cat and dog images. This step helps in understanding the data visually.
