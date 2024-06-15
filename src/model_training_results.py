"""
Implementation of a generic class that stores all the possible model training results. The evaluation metrics are:
- Best Hyperparameters
- Average Accuracy
- Average Recall
- Average Precision
- Average F1-Score
- Standard Deviation of Accuracy

Author
------
@rkalai, rishabh.kalai@unb.ca
"""

# Import the necessary libraries
# 3rd party libraries
import numpy as np
from functools import cached_property


class ModelTrainingResults:
    def __init__(
        self,
        model=None,
        model_name: str = None,
        hyperparameters: dict = None,
        accuracies: list = None,
        recalls: list = None,
        precisions: list = None,
        f1_scores: list = None,
    ):
        """
        Initialize the ModelTrainingResults class.

        Parameters
        ----------
        model : object
            The model object.
        model_name : str
            The model name.
        hyperparameters : dict
            The dictionary of hyperparameters.
        accuracies : list
            The list of accuracies.
        recalls : list
            The list of recalls.
        precisions : list
            The list of precisions.
        f1_scores : list
            The list of f1-scores.
        """
        self.model = model
        self.model_name = model_name
        self.hyperparameters = hyperparameters
        self.accuracies = accuracies
        self.recalls = recalls
        self.precisions = precisions
        self.f1_scores = f1_scores

    @cached_property
    def average_accuracy(self):
        """
        Compute the average accuracy.
        """
        return round(np.mean(self.accuracies), 2)

    @cached_property
    def average_recall(self):
        """
        Compute the average recall.
        """
        return round(np.mean(self.recalls), 2)

    @cached_property
    def average_precision(self):
        """
        Compute the average precision.
        """
        return round(np.mean(self.precisions), 2)

    @cached_property
    def average_f1_score(self):
        """
        Compute the average f1-score.
        """
        return round(np.mean(self.f1_scores), 2)

    @cached_property
    def std_dev_accuracy(self):
        """
        Compute the standard deviation of accuracy.
        """
        return round(np.std(self.accuracies), 2)
