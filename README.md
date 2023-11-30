 # Cat-Dog Classifier Using Neural Networks #

## Overview:
This project involves developing a neural network model to classify creatures as either cats or dogs. Utilizing features like weight, height, fur color, and leg length, the model offers a practical approach to understanding and implementing basic machine learning techniques with TensorFlow and Keras in Python.

## Key Objectives:
- To demonstrate the process of creating a neural network for binary classification.
- To provide hands-on experience with data preprocessing, including normalization and one-hot encoding.
- To showcase the use of TensorFlow and Keras for building and training neural network models.

## Dataset:
The model is trained on a synthetic dataset consisting of 1,000 samples. Each sample includes features such as weight, height, fur color, and leg length, along with a label indicating whether the creature is a cat or a dog.

## Model Architecture:
- Input Layer: 5 neurons (corresponding to the number of features).
- Two Hidden Layers: Each with 10 neurons and ReLU activation functions.
- Output Layer: Single neuron with a sigmoid activation function for binary classification.
- Constraints: Max norm constraints applied to hidden layers for regularization.

## Tools and Libraries:
- Python: For overall programming.
- NumPy: For numerical operations and data manipulation.
- Pandas: For data handling and loading.
- TensorFlow and Keras: For building and training the neural network.
- Scikit-Learn: For data preprocessing and splitting.

## Steps:
- Data Preparation: Load the dataset, normalize features, and encode categorical variables.
- Building the Model: Define the neural network structure using Keras.
- Training: Fit the model to the training data.
- Evaluation and Testing: Assess model performance on a separate test dataset.
- Prediction: Demonstrate how to use the model to classify new samples.

## Usage:
- Clone the repository.
- Install required dependencies.
- Run the provided Jupyter notebook for an interactive experience, or execute the Python script for a direct demonstration.

## Future Scope:
- Experimentation with different network architectures and hyperparameters.
- Application of the model to real-world datasets.
- Integration of more complex features and image data for advanced classification tasks.

## Conclusion:
This project serves as an introductory guide to neural networks and classification problems in machine learning, making it ideal for educational purposes and as a baseline for more complex projects.
