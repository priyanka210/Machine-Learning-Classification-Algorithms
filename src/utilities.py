"""
Implementation of the Utilities module. This module contains the mathematical and data manipulation functions.
This module is responsible for providing the mathematical functions and data manipulation functions used in the machine learning models.

Author
------
@rkalai, rishabh.kalai@unb.ca
"""

# Import the necessary libraries
# 3rd party libraries
import numpy as np

# INDEX OF FUNCTIONS:
# 1. MATHEMATICAL FUNCTIONS
#     1.1 softmax
#     1.2 cross_entropy_loss
#     1.3 derivative_cross_entropy_softmax
#     1.4 mean_squared_error
#     1.5 euclidean_distance
#     1.6 manhattan_distance
# 2. DATA MANIPULATION FUNCTIONS
#     2.1 bootstrap_samples


# ------------------------------------ 1. MATHEMATICAL FUNCTIONS ------------------------------------ #
def softmax(x):
    """
    Compute the softmax of vector x.

    The softmax function, also known as softargmax or normalized exponential function,
    is a function that takes as input a vector of K real numbers, and normalizes it into
    a probability distribution consisting of K probabilities. That is, prior to applying
    softmax, some vector components could be negative, or greater than one; and might not
    sum to 1; but after applying softmax, each component will be in the interval (0,1),
    and the components will add up to 1, so that they can be interpreted as probabilities.
    Furthermore, the larger input components will correspond to larger probabilities.

    Softmax is often used in neural networks, to map the non-normalized output of a network
    to a probability distribution over predicted output classes.

    Parameters:
    x (numpy array): Input Vector or Matrix.

    Returns:
    numpy array: Softmax of the input vector.
    """
    # Softmax Formula: e^x / sum(e^x)
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_loss(y_pred, y_true):
    """
    Compute the cross-entropy loss.

    Cross-entropy loss, or log loss, measures the performance of a classification model
    whose output is a probability value between 0 and 1. Cross-entropy loss increases as
    the predicted probability diverges from the actual label. It is used in machine learning
    and statistics as a measure of difference between two probability distributions.

    Parameters:
    y_pred (numpy array): Predicted values.
    y_true (numpy array): True values.

    Returns:
    float: Cross entropy loss.
    """
    m = y_true.shape[0]
    log_y_pred = np.log(y_pred)
    product = y_true * log_y_pred
    epsilon = 1e-12
    sum_product = np.sum(product + epsilon)
    loss = -sum_product / m
    return loss


def derivative_cross_entropy_softmax(y_pred, y_true):
    """
    Compute the derivative of the cross-entropy loss with respect to softmax.

    This function calculates the gradient of the cross-entropy loss with respect to the
    softmax function, which is used in the backpropagation process of training a neural
    network. The derivative is simply the difference between the predicted and true values.

    Parameters:
    y_pred (numpy array): Predicted values.
    y_true (numpy array): True values.

    Returns:
    numpy array: Derivative of the cross-entropy loss with respect to softmax.
    """
    derivative = y_pred - y_true
    return derivative


def mean_squared_error(y_pred, y_true):
    """
    Compute the mean squared error.

    Mean squared error (MSE) is a common loss function used for regression problems.
    It calculates the average squared difference between the predicted and true values,
    giving a measure of prediction error. The squaring ensures that larger errors are
    more significant than smaller ones.

    Parameters:
    y_pred (numpy array): Predicted values.
    y_true (numpy array): True values.

    Returns:
    float: Mean squared error.
    """
    diff = y_pred - y_true
    squared_diff = np.square(diff)
    mse = np.mean(squared_diff)
    return mse


def euclidean_distance(point1, point2):
    """
    Compute the Euclidean distance between two points.

    The Euclidean distance, also known as the L2 norm or 2-norm, is a measure of the straight line distance between two points in a space.
    It is calculated as the square root of the sum of the squared differences between the corresponding elements of the two vectors.

    This function uses NumPy's vectorized operations to perform the calculations, which is faster than using Python's built-in loops.

    Parameters:
    point1 (numpy array): The first point in the space.
    point2 (numpy array): The second point in the space.

    Returns:
    float: The Euclidean distance between point1 and point2.
    """
    return np.sqrt(np.sum((point1 - point2) ** 2))


def manhattan_distance(point1, point2):
    """
    Compute the Manhattan distance between two points.

    The Manhattan distance, also known as the L1 norm or 1-norm, is a measure of the distance between two points in a grid-based system (like a chessboard or city blocks).
    It is calculated as the sum of the absolute differences between the corresponding elements of the two vectors.

    This function uses NumPy's vectorized operations to perform the calculations, which is faster than using Python's built-in loops.

    Parameters:
    point1 (numpy array): The first point in the space.
    point2 (numpy array): The second point in the space.

    Returns:
    float: The Manhattan distance between point1 and point2.
    """
    return np.sum(np.abs(point1 - point2))


# ------------------------------------ 1. END OF MATHEMATICAL FUNCTIONS ------------------------------------ #
# ---------------------------------------------------------------------------------------------------------- #

# ------------------------------------ 2. DATA MANIPULATION FUNCTIONS -------------------------------------- #


def bootstrap_samples(X, y):
    """
    Bootstrapping is the process of sampling with replacement. This function returns the bootstrapped samples of the dataset.
    This usually includes a random subset of the rows of the dataset. The number of samples is the same as the original dataset.
    The reason for the number of samples being the same as the original dataset is to ensure that the Random Forest has the same number of samples as the original dataset.
    The importance of bootstrapping is to ensure that the Random Forest has a diverse set of samples to train on.

    Parameters
    ----------
    X (np.ndarray): The features of the dataset.
    y (np.ndarray): The target variable of the dataset.

    Returns
    -------
    X_bootstrap_samples (np.ndarray): The bootstrapped samples of the features.
    y_bootstrap_samples (np.ndarray): The bootstrapped samples of the target variable.
    """
    n_samples = X.shape[0]
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    X_bootstrap_samples = X[indices]
    y_bootstrap_samples = y[indices]
    return X_bootstrap_samples, y_bootstrap_samples


# -------------------------------- 2. END OF DATA MANIPULATION FUNCTIONS ----------------------------------- #
# ---------------------------------------------------------------------------------------------------------- #
