"""
Implementation of ANN (Artificial Neural Network) model. The model is a Multi-Layer Perceptron (MLP) model, and is
implemented from scratch.

Author
------
@rkalai, rishabh.kalai@unb.ca
"""

# 3rd party libraries
import numpy as np
from sklearn.metrics import accuracy_score


# Import the necessary libraries
# Custom libraries
import src.constants as CONSTANTS
from src.utilities import softmax, cross_entropy_loss, derivative_cross_entropy_softmax

# GLOBAL SETTINGS
# Set Seed for reproducibility
np.random.seed(CONSTANTS.RANDOM_STATE)


class ANN:
    def __init__(
        self,
        input_size: int = None,
        hidden_size: int = CONSTANTS.HIDDEN_SIZE,
        output_size: int = None,
        learning_rate: float = CONSTANTS.LEARNING_RATE,
        epochs: int = CONSTANTS.EPOCHS,
    ):
        """
        Initialize the ANN model with the given parameters.

        The ANN model is a Multi-Layer Perceptron (MLP) model with one hidden layer.

        Parameters:
        input_size (int): The number of input features.
        hidden_size (int): The number of neurons in the hidden layer.
        output_size (int): The number of output classes.
        learning_rate (float): The learning rate for the gradient descent optimization.
        epochs (int): The number of epochs for training the model.

        Returns:
        None
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def feedforward(self, X):
        """
        Perform the feedforward step of the ANN model.

        The feedforward step involves passing the input data through the network and calculating the output.

        Parameters:
        X (numpy array): The input data.

        Returns:
        numpy array: The output of the network.
        """
        # Calculate the output of the hidden layer
        self.hidden_layer_output = (
            np.dot(X, self.weights_input_hidden) + self.biases_input_hidden
        )
        # Apply the ReLU activation function
        self.hidden_layer_output = np.maximum(0, self.hidden_layer_output)
        # Calculate the output of the output layer
        self.output_layer_output = (
            np.dot(self.hidden_layer_output, self.weights_hidden_output)
            + self.biases_hidden_output
        )
        # Apply the softmax activation function
        self.output_layer_output = softmax(self.output_layer_output)
        return self.output_layer_output

    def backpropagation(self, X, y):
        """
        Perform the backpropagation step of the ANN model.

        The backpropagation step involves calculating the error of the model and updating the weights and biases accordingly.

        Parameters:
        X (numpy array): The input data.
        y (numpy array): The true output data.

        Returns:
        None
        """
        # Calculate the output of the network
        y_pred = self.feedforward(X)
        # Calculate the error of the output layer
        error = derivative_cross_entropy_softmax(y_pred, y)
        # Calculate the gradients for the weights and biases of the output layer
        weights_hidden_output_update = (
            np.dot(self.hidden_layer_output.T, error) / X.shape[0]
        )
        biases_hidden_output_update = np.sum(error, axis=0, keepdims=True) / X.shape[0]
        # Calculate the error of the hidden layer
        error_hidden_layer = np.dot(error, self.weights_hidden_output.T)
        error_hidden_layer[self.hidden_layer_output <= 0] = 0
        # Calculate the gradients for the weights and biases of the hidden layer
        weights_input_hidden_update = np.dot(X.T, error_hidden_layer) / X.shape[0]
        biases_input_hidden_update = (
            np.sum(error_hidden_layer, axis=0, keepdims=True) / X.shape[0]
        )
        # Update the weights and biases of the output layer
        self.weights_hidden_output -= self.learning_rate * weights_hidden_output_update
        self.biases_hidden_output -= self.learning_rate * biases_hidden_output_update
        # Update the weights and biases of the hidden layer
        self.weights_input_hidden -= self.learning_rate * weights_input_hidden_update
        self.biases_input_hidden -= self.learning_rate * biases_input_hidden_update

    def train(self, X, y, epochs):
        """
        Train the ANN model with the given input and output data for a certain number of epochs.

        Parameters:
        X (numpy array): The input data.
        y (numpy array): The true output data.
        epochs (int): The number of epochs for training the model.

        Returns:
        None
        """
        for epoch in range(epochs):
            self.backpropagation(X, y)

    def fit(self, X, y=None):
        """
        Fit the ANN model to the given input and output data.

        This function is a wrapper for the train function.

        Parameters:
        X (numpy array): The input data.
        y (numpy array): The true output data.

        Returns:
        self: The instance of the model.
        """
        # Initialize the input and output sizes, and the weights and biases
        self.input_size = X.shape[1]
        self.output_size = y.shape[1]
        # Initialize the weights and biases with random values
        self.weights_input_hidden = np.random.randn(
            self.input_size, self.hidden_size
        ) * np.sqrt(2.0 / self.input_size)
        self.biases_input_hidden = np.zeros((1, self.hidden_size))
        self.weights_hidden_output = np.random.randn(
            self.hidden_size, self.output_size
        ) * np.sqrt(2.0 / self.hidden_size)
        self.biases_hidden_output = np.zeros((1, self.output_size))
        # Train the model
        self.train(X, y, self.epochs)
        return self

    def predict(self, X):
        """
        Predict the output for the given input data.

        Parameters:
        X (numpy array): The input data.

        Returns:
        numpy array: The predicted output data.
        """
        y_pred = self.feedforward(X)
        return (y_pred > 0.5).astype(int)

    def score(self, X, y):
        """
        Calculate the accuracy of the model for the given input and output data.

        Parameters:
        X (numpy array): The input data.
        y (numpy array): The true output data.

        Returns:
        float: The accuracy of the model.
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def get_params(self, deep=True):
        """
        Get the parameters of the model.

        Parameters:
        deep (bool): Whether to return the parameters of the sub-estimators.

        Returns:
        dict: The parameters of the model.
        """
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
        }

    def set_params(self, **parameters):
        """
        Set the parameters of the model.

        Parameters:
        parameters (dict): The new parameters of the model.

        Returns:
        self: The instance of the model.
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
