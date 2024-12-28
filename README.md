# Skin Cancer Classification with CNN and Tabular Data Integration

This repository contains a comprehensive deep learning project for classifying skin cancer using the HAM10000 dataset, which combines image-based and tabular data inputs to achieve accurate predictions. The project involves extensive preprocessing, model building, and deployment.

## Introduction

Skin cancer is one of the most common cancers worldwide, and early detection is crucial for effective treatment. This project aims to develop a robust classification model that leverages both dermatoscopic images and tabular metadata to predict the type of skin lesion accurately. The integration of image and tabular data provides a richer context for making predictions, enabling the model to consider factors such as lesion localization, patient age, and diagnostic methods.

## Motivation

The choice of this project stems from the following reasons:

- **Real-World Impact**: Skin cancer detection is a critical medical problem with a direct impact on patients' lives. Developing an AI model for early detection can significantly assist dermatologists in diagnosing and prioritizing cases.
- **Dataset Diversity**: The HAM10000 dataset offers a unique opportunity to work with multimodal data, combining images and metadata. This presents a challenging yet rewarding task of integrating different data modalities.
- **Skill Development**: This project provides hands-on experience in tackling issues like class imbalance, data augmentation, and model deployment, which are prevalent in real-world machine learning applications.

By choosing this project, the goal is to contribute to the field of medical AI while simultaneously enhancing skills in data preprocessing, model design, and deployment.

## Table of Contents

1. Dataset
2. Exploratory Data Analysis
3. Preprocessing
4. Model Architecture
5. Training and Evaluation
6. Deployment
7. Challenges and Solutions
8. Usage Instructions
9. Dependencies

## Dataset

The project uses the HAM10000 dataset from Kaggle: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

### Dataset Highlights:

- **10,015 dermatoscopic images**.
- Metadata includes lesion type, localization, age, gender, and diagnostic method.

### Categories:

- Actinic keratoses (akiec)
- Basal cell carcinoma (bcc)
- Benign keratosis (bkl)
- Dermatofibroma (df)
- Melanoma (mel)
- Melanocytic nevi (nv)
- Vascular lesions (vasc).

Visualizations of the dataset's insights are available in the repository (e.g., age distribution, lesion types, gender distribution).

## Exploratory Data Analysis

Extensive visualizations were performed to understand the dataset better. Below are key plots included in the repository:

- Age distribution by lesion type.
- Diagnosis by gender.
- Localization vs. diagnosis heatmap.
- Lesion type and diagnosis method distributions.

## Preprocessing

The preprocessing steps included:

- **Handling Missing Values**: Filling missing age values with the mean.
- **Feature Engineering**: Dropped unnecessary columns and encoded categorical data (e.g., gender, localization).
- **Normalization**: Applied MinMaxScaler to normalize tabular data.
- **Data Augmentation**: To address class imbalance:
  - **Techniques**: Rotation, horizontal/vertical flipping, noise addition, blurring, color jitter, and normalization.
  - **Tools**: Albumentations library.

## Model Architecture

The model combines convolutional neural networks (CNN) for image data and fully connected layers for tabular data:

### Image Input Branch:

The image input branch processes 28x28x3 RGB images through a series of convolutional and pooling layers to extract features. The final feature map is flattened and passed through dense layers to produce a feature vector representation of the image.

### Tabular Input Branch:

The tabular input branch handles four numerical features through dense layers to extract meaningful representations.

### Combined Model:

The outputs of the image and tabular branches are concatenated and passed through additional dense layers for joint learning. The final layer outputs probabilities for seven skin lesion classes using the softmax activation function.

### Model Features:

- Early stopping.
- Model checkpointing.
- Learning rate reduction on plateau.

## Training and Evaluation

The model was trained for 100 epochs with the following results:

- **Train Accuracy**: 76.81%
- **Validation Accuracy**: 72.49%
- **Test Accuracy**: 71.64%

Evaluation metrics include:

- Confusion matrix.
- Precision, Recall, and F1-Score.

Visualizations include training/validation loss and accuracy curves.

## Deployment

The model was deployed as a web application with a simple user interface:

- **Frontend**: Flask, Python.
- **Features**: Upload an image for skin lesion prediction.

Screenshots of the web app are included in the repository.

## Challenges and Solutions

### Class Imbalance:

- **Challenge**: Over-representation of certain lesion types.
- **Solution**: Data augmentation and oversampling of minority classes.

### Overfitting:

- **Challenge**: Small dataset size leading to overfitting.
- **Solution**: Regularization, dropout, and early stopping.

## Usage Instructions

1. Clone the repository.
2. Install dependencies using `requirements.txt`.
3. Train the model or use the pretrained weights provided.
4. Run the web application using the provided Flask server script.

## Dependencies

The project uses the following libraries and frameworks with their minimum versions:

### Python Libraries:

- `pandas>=1.3.0`
- `numpy>=1.21.0`
- `matplotlib>=3.4.0`
- `seaborn>=0.11.0`
- `Pillow>=8.2.0`
- `albumentations>=1.1.0`
- `scikit-learn>=0.24.0`
- `tensorflow>=2.5.0`
- `flask>=2.0.0`

### Built-In Libraries:

- `collections`
- `warnings`

For any questions or issues, please create an issue in this repository or contact with me.
