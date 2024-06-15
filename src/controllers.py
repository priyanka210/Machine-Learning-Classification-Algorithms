"""
Implementation of the Controller Layer for the Machine Learning Model Training and Evaluation Pipeline.
This module is responsible for orchestrating the training and evaluation of the machine learning models.

Author
------
@rkalai, rishabh.kalai@unb.ca
"""

# Import the necessary libraries
# 3rd party libraries
import concurrent.futures

# Custom libraries
import src.constants as CONSTANTS
import src.data_preprocessing as PREPROCESSING
import src.model_evaluation as EVALUATION
import src.model_training_results as RESULTS

# Model Libraries
from src.adaboost import AdaBoostMultiClass
from src.ann import ANN
from src.knn import KNN
from src.random_forest import RandomForest


def perform_model_training_and_validation(
    model=None,
    dataset: list = None,
    hyperparameters_grid: dict = None,
    hyperparameters: dict = None,
):
    """
    Perform the model training and validation using Cross-Validation. This function trains the model using:
    - Hyperparameter Search (if hyperparameters_grid is provided)
    - Cross-Validation
    - Model Performance Evaluation

    Parameters
    ----------
    model : object
        The model object.
    dataset : list
        The dataset object.
    hyperparameters_grid : dict
        The dictionary of hyperparameters.
    hyperparameters : dict
        The dictionary of hyperparameters.

    Returns
    -------
    object
        The model training results.
    """
    X, y, feature_headers, target_headers, dataset_name = dataset
    # Initialize the model
    if hyperparameters_grid:
        best_model, best_hyperparameters, accuracy = (
            EVALUATION.hyperparameter_search_and_train(
                model=model, param_distributions=hyperparameters_grid, X=X, y=y
            )
        )
    else:
        best_model = model
        best_hyperparameters = hyperparameters
    # Train the model using Cross-Validation
    accuracies, recalls, precisions, f1_scores = EVALUATION.evaluate_model_performance(
        best_model, X, y
    )
    # Obtain Model Name as defined by the Class Name
    model_name = model.__class__.__name__
    # Create a Model Training Results Object
    model_results = RESULTS.ModelTrainingResults(
        model=best_model,
        model_name=model_name,
        hyperparameters=best_hyperparameters,
        accuracies=accuracies,
        recalls=recalls,
        precisions=precisions,
        f1_scores=f1_scores,
    )
    return model_results


def train_and_validate_model(
    model_class,
    dataset: list = None,
    hyperparameters_dict: dict = None,
    hyperparameters_grid_dict: dict = None,
    hyperparameter_search: bool = False,
):
    """
    Train and Validate a Machine Learning Model on the given dataset from the UCIML Repository.

    Parameters
    ----------
    model_class : object
        The model class.
    dataset : list
        The dataset object.
    hyperparameters_dict : dict
        The dictionary of hyperparameters.
    hyperparameters_grid_dict : dict
        The dictionary of hyperparameters grid search spaces.
    hyperparameter_search : bool
        A boolean flag to indicate if hyperparameter search should be performed.

    Returns
    -------
    object
        The model training results.
    """
    hyperparameters_grid = None
    if not hyperparameter_search:
        # This is in the case that hyperparameters are not provided and the model follows the default hyperparameters as is the case in KNN
        if hyperparameters_dict is not None:
            model = model_class(**hyperparameters_dict)
        else:
            model = model_class()
    else:
        hyperparameters_grid = hyperparameters_grid_dict
        model = model_class()
    model_results = perform_model_training_and_validation(
        model=model,
        dataset=dataset,
        hyperparameters_grid=hyperparameters_grid,
        hyperparameters=hyperparameters_dict,
    )
    return model_results


def train_and_validate_all_models(ID: str = None):
    """
    Train and Validate all the Machine Learning Models on the given dataset from the UCIML Repository.

    Parameters
    ----------
    ID : str
        The ID of the dataset from the UCIML Repository.

    Returns
    -------
    list
        The list of model results obtained from training and validating each of the models.
    """
    # Define the list of model results, and a dictionary of model accuracies for each of the models trained
    model_results = []
    model_accuracies = {}
    model_accuracy_distributions = {}
    model_class_to_hyperparameters = {
        RandomForest: (
            CONSTANTS.RANDOM_FOREST_HYPERPARAMETERS[ID],
            CONSTANTS.RANDOM_FOREST_HYPERPARAMETER_GRID,
        ),
        AdaBoostMultiClass: (
            CONSTANTS.ADABOOST_HYPERPARAMETERS[ID],
            CONSTANTS.ADABOOST_HYPERPARAMETER_GRID,
        ),
        KNN: (None, None),
        ANN: (CONSTANTS.ANN_HYPERPARAMETERS[ID], CONSTANTS.ANN_HYPERPARAMETER_GRID),
    }
    # Define a dictionary mapping the model classes to their respective encoding schemes
    model_class_to_encoding_scheme = {
        RandomForest: CONSTANTS.RANDOM_FOREST_DATA_ENCODING_SCHEME,
        AdaBoostMultiClass: CONSTANTS.ADABOOST_DATA_ENCODING_SCHEME,
        KNN: CONSTANTS.KNN_DATA_ENCODING_SCHEME,
        ANN: CONSTANTS.ANN_DATA_ENCODING_SCHEME,
    }
    # Iterate over the dictionary and train the models
    for model_class, encoding_scheme in model_class_to_encoding_scheme.items():
        # Select the dataset from the UCIML Repository with the appropriate encoding scheme
        X, y, feature_headers, target_headers, dataset_name = (
            PREPROCESSING.select_uci_dataset(
                dataset_id=ID,
                categorical_encoding_scheme=encoding_scheme,
                scale_data=True,
            )
        )
        dataset = [X, y, feature_headers, target_headers, dataset_name]
        # Create a ThreadPoolExecutor and train the model
        with concurrent.futures.ThreadPoolExecutor() as executor:
            hyperparameters_dict, hyperparameters_grid_dict = (
                model_class_to_hyperparameters[model_class]
            )
            future_model = executor.submit(
                train_and_validate_model,
                model_class,
                dataset,
                hyperparameters_dict,
                hyperparameters_grid_dict,
                False,
            )
            model_results_obj = future_model.result()
            model_results.append(model_results_obj)
            model_accuracies[model_results_obj.model_name] = (
                model_results_obj.average_accuracy
            )
            model_accuracy_distribution_plot_title = f"{model_results_obj.model_name} Accuracy: {dataset_name}\n Avg. Accuracy: {model_results_obj.average_accuracy}% | Std. Dev Accuracy: {model_results_obj.std_dev_accuracy}%"
            model_accuracy_distributions[model_accuracy_distribution_plot_title] = (
                model_results_obj.accuracies
            )
    # Plot the Model Accuracy Distributions
    EVALUATION.facet_histogram(
        plot_data=model_accuracy_distributions,
        plot_title=f"Model Accuracy Distributions: {dataset_name}",
        horizontal_axis_label="Accuracy (%)",
        vertical_axis_label="Frequency",
    )
    # Sort the accuracies in descending order
    model_accuracies = dict(
        sorted(model_accuracies.items(), key=lambda item: item[1], reverse=True)
    )
    # Plot a bar chart with the average accuracies of each of the models
    EVALUATION.plot_bar_chart(
        plot_data=model_accuracies,
        horizontal_axis_label="Model",
        vertical_axis_label="Average Accuracy (%)",
        plot_title=f"Average Accuracies of Models: {dataset_name}",
        show_data_values=True,
    )
    return model_results
