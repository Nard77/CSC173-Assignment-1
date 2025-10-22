# CSC173 Activity 01 - Neural Network from Scratch

**Date:** October 22, 2025  
**Team:** [Group CSC173
	   Members:
		Sistona, Lavigne
		Garbanzos, Louise
		Dadula, Bernard]

## Project Overview

This project implements a simple neural network for binary classification using breast cancer diagnostic data. The network was built entirely from scratch using Python, NumPy, and Matplotlib, with no external machine learning libraries such as scikit-learn or TensorFlow.

The goal of this project is to understand the core mechanics of a neural network — including forward propagation, loss computation, backpropagation, gradient descent, and model evaluation — by implementing each step manually.

## Data Preparation

We used the Breast Cancer Wisconsin Diagnostic dataset from the UCI Machine Learning Repository

Each sample in the dataset describes characteristics of a breast mass and whether it is malignant (cancerous) or benign (non-cancerous).

For this project, we only used two features to keep the neural network simple and easy to visualize:

-Mean Radius – the average size of the cell nuclei

-Mean Texture – the variation in cell texture

Before training, the data was cleaned and prepared as follows:
The target labels were converted to numbers: Malignant = 1, Benign = 0
Each feature was normalized (scaled) so that the values are within a similar range — this helps the network learn faster
The dataset was split into training (80%) and testing (20%) parts to evaluate the model’s accuracy later

## Network Architecture

The neural network used in this project has the following structure:

-Input Layer: 2 neurons (for the two selected features: mean radius and mean texture)

-Hidden Layer: 3 neurons using the Tanh activation function

-Output Layer: 1 neuron with a Sigmoid activation function for binary classification

The network predicts whether a tumor is malignant (1) or benign (0) based on the two input features.

## Implementation Details

-Weight Initialization: All weights and biases were initialized randomly with small values.

-Activation Functions: The hidden layer uses Tanh, while the output layer uses Sigmoid to produce values between 0 and 1.

-Forward Propagation: Each input passes through the layers to compute predictions.

-Loss Function: The model uses Mean Squared Error (MSE) to measure prediction error.

-Backpropagation: Gradients of weights and biases are calculated using the chain rule.

-Gradient Descent: Parameters are updated using a fixed learning rate to reduce loss after each iteration.

-Training: The network is trained for 500 epochs to minimize loss.

-Testing: After training, the model’s accuracy is evaluated using unseen test data.

## Results & Visualization

The neural network was trained for 500 epochs, and the loss consistently decreased throughout training, showing that the model successfully learned from the data.

- Training Loss (MSE): 
The loss curve showed a steady downward trend, confirming that the model’s parameters were updated correctly during backpropagation and gradient descent.

After training, the network was evaluated on the test set and achieved a test accuracy of 82.46%, meaning it correctly classified roughly 8 out of 10 cases.

-Sample Predictions:
The results show that even with only two input features and a small hidden layer, the network was able to learn meaningful patterns from the data and produce reliable predictions for classifying breast cancer cases.



## Team Collaboration

Each member contributed to different components of the network:
- Weight and bias initialization
	-Lavigne focused on designing and coding the initialization process for weights and biases, ensuring they were randomly and appropriately initialized to promote effective learning and prevent issues such as vanishing gradients.

- Loss function implementation
	-Louise was responsible for defining and implementing the loss function, which measured the model's prediction errors. They ensured that the chosen loss function aligned with the dataset and overall training objectives.

- Training loop and visualization
	-While, I, Bernard, developed the training loop, integrating all components into an iterative process for updating parameters over multiple epochs. They also implemented visualization tools, such as loss curves and decision boundary plots, to track training progress and model performance.

## How to Run

1. Clone the GitHub repository:
   git clone https://github.com/Nard77/CSC173-GroupAssignment-1

2. Open the Jupyter notebook or Colab file.
3. Run all cells sequentially.
4. Explore training loss plot and decision boundary visualizations.

## Summary

This activity provided hands-on experience in building a neural network from scratch using Python and NumPy. The group implemented data preprocessing, forward and backward propagation, and visualized the model’s learning progress. The project strengthened understanding of how neural networks learn and adapt through gradient-based optimization.

Video: https://drive.google.com/file/d/1exhJZg0Pj5CPZdVdxY-naJSwT08K2BjZ/view?usp=sharing
