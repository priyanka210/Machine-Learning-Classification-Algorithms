"""
This module contains functions to evaluate the performance of machine learning models. The functions include:
- Hyperparameter Search and Training
- Evaluate Fold
- Evaluate Model Performance
- Plots to Evaluate Model Performance

Author:
@rkalai
"""

# Plotting Libraries
import matplotlib.pyplot as plt

# 3rd party libraries
import numpy as np
import seaborn as sns
from sklearn.base import clone
from sklearn.experimental import enable_halving_search_cv
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from skmultilearn.model_selection import IterativeStratification

# Import the necessary libraries
# Custom libraries
import src.constants as CONSTANTS

# GLOBAL SETTINGS
# Set Seed for reproducibility
np.random.seed(CONSTANTS.RANDOM_STATE)
# Call the enable_halving_search_cv to enable the HalvingRandomSearchCV class
_ = enable_halving_search_cv

# INDEX OF FUNCTIONS
# 1. MODEL EVALUATION FUNCTIONS
#     1.1 hyperparameter_search_and_train
#     1.2 evaluate_fold
#     1.3 evaluate_model_performance
# 2. PLOTTING FUNCTIONS
#     2.1 plot_confusion_matrix
#     2.2 plot_histogram
#     2.3 plot_bar_chart


# -------------------------------------- 1. MODEL EVALUATION FUNCTIONS ---------------------------------------- #
def hyperparameter_search_and_train(
    model=None, param_distributions: dict = None, X: np.array = None, y: np.array = None
):
    """
    Perform a hyperparameter search and train the model. This function uses the HalvingRandomSearchCV class.

    Parameters
    ----------
    model : object
        The model object.
    param_distributions : dict
        The dictionary of parameter distributions.
    X : array-like
        The input features.
    y : array-like
        The target labels.

    Returns
    -------
    object
        The best model.
    dict
        The best hyperparameters.
    float
        The best accuracy.
    """
    # Create an instance of the HalvingRandomSearchCV class
    # This class is used to perform a randomized search over the hyperparameters of the model
    # It is experimental, but is more efficient than the RandomizedSearchCV class
    rsh = HalvingRandomSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        factor=2,
        random_state=0,
    )
    rsh.fit(X, y)
    best_hyper_parameters = rsh.best_params_
    best_model = rsh.best_estimator_
    best_accuracy = rsh.best_score_
    return best_model, best_hyper_parameters, best_accuracy


def evaluate_fold(
    ml_model=None,
    X_train: np.array = None,
    X_test: np.array = None,
    y_train: np.array = None,
    y_test: np.array = None,
):
    """
    Evaluate the model performance for a fold in the cross-validation process.

    Parameters
    ----------
    ml_model : object
        The machine learning model.
    X_train : array-like
        The input features of the training set.
    X_test : array-like
        The input features of the test set.
    y_train : array-like
        The target labels of the training set.
    y_test : array-like
        The target labels of the test set.

    Returns
    -------
    dict
        The classification report.
    """
    model = clone(ml_model)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    if y_test.ndim > 1:
        # Convert one-hot encoded vectors to labels for classification report
        y_test = np.argmax(y_test, axis=1)
    if predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=1)
    # Get the classification report for the model, with accuracy, precision, recall, and F1-Score
    report = classification_report(y_test, predictions, output_dict=True)
    return report


def evaluate_model_performance(
    ml_model=None,
    X: np.array = None,
    y: np.array = None,
    n_splits: int = CONSTANTS.N_SPLITS,
    n_repeats: int = CONSTANTS.N_REPEATS,
):
    """
    Evaluate the model performance using Cross-Validation.

    Parameters
    ----------
    ml_model : object
        The machine learning model.
    X : array-like
        The input features.
    y : array-like
        The target labels.
    n_splits : int
        The number of splits.
    n_repeats : int
        The number of repeats.

    Returns
    -------
    list
        The classification reports.
    """
    one_hot_encoded_flag = y.ndim > 1
    if one_hot_encoded_flag:
        # Use IterativeStratification for multi-label classification
        skf = IterativeStratification(n_splits=n_splits, order=1)
        reports = [
            evaluate_fold(
                ml_model, X[train_index], X[test_index], y[train_index], y[test_index]
            )
            for train_index, test_index in skf.split(X, y)
        ]
    else:
        skf = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=CONSTANTS.RANDOM_STATE
        )
        reports = [
            evaluate_fold(
                ml_model, X[train_index], X[test_index], y[train_index], y[test_index]
            )
            for train_index, test_index in skf.split(X, y)
        ]
    # Initialize lists to store average metrics
    accuracies, recalls, precisions, f1_scores = [], [], [], []
    for report in reports:
        accuracies.append(report["accuracy"] * 100)
        recalls.append(report["macro avg"]["recall"] * 100)
        precisions.append(report["macro avg"]["precision"] * 100)
        f1_scores.append(report["macro avg"]["f1-score"] * 100)
    return accuracies, recalls, precisions, f1_scores


# ------------------------------------- END OF MODEL EVALUATION FUNCTIONS -------------------------------- #
# -------------------------------------------------------------------------------------------------------- #


# -------------------------------------- 2. PLOTTING FUNCTIONS ------------------------------------------- #


def plot_confusion_matrix(
    true_labels: np.array = None,
    predicted_labels: np.array = None,
    class_names: list = None,
    title: str = None,
):
    """
    Plot the confusion matrix as a heatmap.

    Parameters
    ----------
    true_labels : array-like
        The true labels.
    predicted_labels : array-like
        The predicted labels.
    class_names : list
        The names of the target classes.
    title : str
        The title of the plot.
    """
    cm = confusion_matrix(true_labels, predicted_labels)
    fig = plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="g")
    plt.xticks(ticks=np.arange(0.5, len(class_names)), labels=class_names)
    plt.yticks(ticks=np.arange(0.5, len(class_names)), labels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix: {title}")
    plt.show()


def plot_histogram(
    plot_data: list = None,
    horizontal_axis_label: str = "Metric",
    vertical_axis_label: str = "Frequency",
    plot_title: str = None,
    kde: bool = True,
):
    """
    Plot the histogram of the data.

    Parameters
    ----------
    plot_data : list
        The list of data to be plotted.
    horizontal_axis_label : str
        The label of the horizontal axis.
    vertical_axis_label : str
        The label of the vertical axis.
    plot_title : str
        The title of the plot.
    kde : bool
        Whether to show the Kernel Density Estimation.
    """
    # Plot the Histogram of the Data
    fig = plt.figure(figsize=(10, 7))
    sns.histplot(plot_data, kde=kde)
    plt.xlabel(horizontal_axis_label)
    plt.ylabel(vertical_axis_label)
    plt.title(plot_title)
    plt.show()


def plot_bar_chart(
    plot_data: dict = None,
    horizontal_axis_label: str = "Metric",
    vertical_axis_label: str = "Frequency",
    plot_title: str = None,
    show_data_values: bool = False,
):
    """
    Plot the bar chart of the data.

    Parameters
    ----------
    plot_data : dict
        The dictionary of data to be plotted.
    horizontal_axis_label : str
        The label of the horizontal axis.
    vertical_axis_label : str
        The label of the vertical axis.
    plot_title : str
        The title of the plot.
    show_data_values : bool
        Whether to show the data values on top of the bars.
    """
    numerical_values = list(plot_data.values())
    categorical_values = list(plot_data.keys())
    # Adjust the Y-axis range shown if the difference between the minimum and maximum values is less than 10 percent
    max_y_value = max(numerical_values)
    min_y_value = min(numerical_values)
    range_y_value = max_y_value - min_y_value
    # The bottom threshold for the change in scale is 10% of the maximum value
    # If the difference between the maximum and minimum values is less than this threshold, adjust the scale
    change_scale_threshold = 0.1 * max_y_value
    fig = plt.figure(figsize=(10, 7))
    if range_y_value < change_scale_threshold:
        # Adjust the Y-axis to enhance the visibility of the difference in the values
        plt.ylim(0.95 * min_y_value, 1.05 * max_y_value)
    sns.barplot(x=categorical_values, y=numerical_values)
    plt.xlabel(horizontal_axis_label)
    plt.ylabel(vertical_axis_label)
    plt.title(plot_title)
    if show_data_values:
        for index, value in enumerate(numerical_values):
            plt.text(
                index,
                value,
                str(value),
                ha="center",
                fontweight="bold",
                fontsize=14,
            )
    plt.show()


def facet_histogram(
    plot_data: dict = None,
    plot_title: str = None,
    horizontal_axis_label: str = None,
    vertical_axis_label: str = None,
    kde: bool = True,
):
    """
    Plot a Facetted Histogram of the Data, with multiple subplots.

    Parameters
    ----------
    plot_data : dict
        The dictionary of data to be plotted.
    plot_title : str
        The title of the plot.
    horizontal_axis_label : str
        The label of the horizontal axis.
    vertical_axis_label : str
        The label of the vertical axis.
    kde : bool
        Whether to show the Kernel Density Estimation.
    """
    # Determine the number of subplots based on the number of keys in the dictionary
    num_subplots = len(plot_data.keys())
    # The number of plots must be even, so if the number of subplots is odd, raise an error
    if num_subplots % 2 != 0:
        raise ValueError("The number of subplots must be even.")
    # Find the number of rows and columns for the subplots
    num_rows = max(1, num_subplots // 2)
    num_columns = min(2, num_subplots)
    # Create the subplots for the data
    fig, axs = plt.subplots(num_rows, num_columns, figsize=(15, 10))
    # Flatten the Axes array
    axs = axs.flatten()
    # Iterate over the dictionary and plot the data in the subplots
    for index, (title, data) in enumerate(plot_data.items()):
        sns.histplot(data, kde=kde, ax=axs[index])
        axs[index].set_title(title)
        axs[index].set_xlabel(horizontal_axis_label)
        axs[index].set_ylabel(vertical_axis_label)
    plt.suptitle(plot_title)
    # Tight layout to prevent overlap of the subplots
    plt.tight_layout()
    plt.show()


# ------------------------------------- END OF PLOTTING FUNCTIONS ---------------------------------------- #
# -------------------------------------------------------------------------------------------------------- #
