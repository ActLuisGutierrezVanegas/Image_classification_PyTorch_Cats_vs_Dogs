# Image_classification_PyTorch_Cats_vs_Dogs

This repository presents a Convolutional Neural Network (CNN) built with PyTorch for binary image classification of cats and dogs.

The project covers the full workflow, including model architecture, training, evaluation, model persistence, model loading, and inference on unseen images.

## Project Objective

The objective of this project is to build a Convolutional Neural Network (CNN) capable of classifying images of cats and dogs and to evaluate its predictive performance on unseen data.

## Dataset

The dataset contains two image classes: cats and dogs. The classes are approximately balanced:

- Cats: 2,403 images (49%)
- Dogs: 2,495 images (51%)

The dataset was split into:

- Training set: 70%
- Validation set: 15%
- Test set: 15%

## Preprocessing

The model was developed and trained in Google Colab using GPU acceleration.

All images were resized to 64 × 64 pixels to reduce computational cost while preserving enough visual information for binary image classification.

The images were then converted into PyTorch tensors with dimensions:

`3 × 64 × 64`

where the three channels represent RGB (Red, Green, Blue).

The dataset was divided into training, validation, and test subsets. PyTorch DataLoaders were used to process the images in mini-batches of 64 samples.

## CNN Architecture

A simple CNN architecture with two convolutional blocks was implemented:

- Conv2D → ReLU → MaxPool
- Conv2D → ReLU → MaxPool
- Flatten
- Linear → 2 output classes

## Training

Two optimizers were compared while keeping the same CNN architecture:

- SGD (Stochastic Gradient Descent)
- Adam

Both configurations used:

- Learning rate: `0.001`
- Epochs: `20`
- Loss function: `Cross Entropy Loss`
- Batch size: `64`

## Evaluation Metrics

The model was evaluated using the following metrics:

- **Accuracy**: Measures the overall proportion of correctly classified images.
- **Precision**: Measures how many images predicted as dogs were actually dogs.
- **Recall**: Measures how many real dog images were correctly identified by the model.
- **F1-score**: Represents the balance between Precision and Recall.

Since the dataset is approximately balanced between cats and dogs, **Accuracy** was selected as the main evaluation metric. The **F1-score** was also considered to evaluate the balance between Precision and Recall.


## Results

The final model achieved the following performance on the test set:

- Accuracy: 74.56%
- Precision: 77.43%
- Recall: 71.50%
- F1-score: 74.35%

The test results are consistent with the performance observed during validation, suggesting that the model generalizes adequately to unseen images.

## Confusion Matrix

The confusion matrix provides a detailed view of the model predictions for each class.

![Confusion Matrix](imagesPrueba/confusion_matrix.png)

Inferencia con imágenes nuevas
- Aquí pondría las capturas que hiciste con los stickers.
- Explicar que el modelo fue guardado en .pth y posteriormente cargado para hacer predicciones sobre imágenes no vistas.

Persistencia del modelo
- Explicar que usaste:

- Estructura del repositorio- Algo como:
- .
├── notebook/
├── models/
├── images/
├── README.md
└── requirements.txt
