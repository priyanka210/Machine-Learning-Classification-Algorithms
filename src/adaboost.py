"""
Implementation of AdaBoost with Tree Stumps from scratch. This module contains the AdaBoostMultiClass class that implements the AdaBoost algorithm for multi-class classification using Tree Stumps as weak learners.

Author
------
@pbhamare, priyanka.bhamare@unb.ca
"""

import numpy as np
from sklearn.metrics import accuracy_score

import src.constants as CONSTANTS


class TreeStump:
    def __init__(self):
        self.threshold = None
        self.feature_index = None
        self.decision_rule = None

    def predict(self, X):
        # Make predictions using the trained tree stump.
        X_column = X[:, self.feature_index]  # Get the feature column
        predictions = np.where(
            X_column < self.threshold,
            self.decision_rule["left"],
            self.decision_rule["right"],
        )
        return predictions

    def fit(self, X, y, weights):
        # Fit the tree stump to the binary labeled data.
        n_samples, n_features = X.shape
        best_err = float("inf")  # Initialize the best error as positive infinity

        # Determine unique classes (binary classification assumed)
        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            raise ValueError(
                "TreeStump supports binary classification. Ensure y contains two unique classes."
            )

        # Iterate over all features and possible thresholds to find the best rule and split point that minimizes error
        for feature_index in range(n_features):
            X_column = X[:, feature_index]
            thresholds = np.unique(X_column)

            # Iterate over all unique values in the feature column
            for threshold in thresholds:
                for left_class in unique_classes:
                    right_class = unique_classes[unique_classes != left_class][
                        0
                    ]  # Pick the other class
                    predictions = np.where(
                        X_column < threshold, left_class, right_class
                    )
                    y = y.reshape(-1)
                    misclassified = predictions != y
                    err = np.sum(weights[misclassified]) / np.sum(weights)

                    if err < best_err:
                        best_err = err
                        self.threshold = threshold
                        self.feature_index = feature_index
                        self.decision_rule = {"left": left_class, "right": right_class}


# Create a custom AdaBoost classifier for multi-class classification using Tree Stumps as weak learners
class AdaBoostMultiClass:
    def __init__(
        self,
        n_estimators=CONSTANTS.ADABOOST_N_ESTIMATORS,
        learning_rate=CONSTANTS.ADABOOST_LEARNING_RATE,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators = []
        self.alphas = []
        self.classes_ = None
        self.is_one_hot_encoded = False

    # Check if the target is one-hot encoded. Assumes y is a NumPy array.
    def _check_one_hot_encoded(self, y):
        if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1):
            return False  # Label encoded
        return True  # One-hot encoded

    # Train the AdaBoost classifier for multi-class classification
    def fit(self, X, y):
        # Determine if y is one-hot encoded and get unique classes
        one_hot_encoded = self._check_one_hot_encoded(y)
        if one_hot_encoded:
            self.classes_ = np.arange(y.shape[1])
            self.is_one_hot_encoded = True
        else:
            self.classes_ = np.unique(y)
            self.is_one_hot_encoded = False

        n_samples, _ = X.shape

        # Train a classifier for each class
        for cls_idx, cls in enumerate(self.classes_):
            W = np.ones(n_samples) / n_samples  # Initialize weights uniformly
            estimators_cls = []
            alphas_cls = []

            # Train n_estimators number of weak classifiers (stumps)
            for _ in range(self.n_estimators):
                stump = TreeStump()  # Initialize a new decision stump
                # Prepare y_binary for the current class vs. rest
                if one_hot_encoded:
                    y_binary = y[:, cls_idx] * 2 - 1  # Convert from [0, 1] to [-1, 1]
                else:
                    y_binary = np.where(y == cls, 1, -1)

                stump.fit(X, y_binary, W)
                predictions = stump.predict(X)

                # Reshape y_binary if it is not one-hot encoded
                if not one_hot_encoded:
                    y_binary = y_binary.reshape(-1)

                # Calculate errors and update weights
                incorrect = predictions != y_binary
                weighted_error = np.dot(W, incorrect) / np.sum(W)

                # Alpha calculation with smoothing to avoid division by zero
                alpha = (
                    self.learning_rate
                    * 0.5
                    * np.log((1.0 - weighted_error) / (max(weighted_error, 1e-10)))
                )

                # Update weights and normalize
                W *= np.exp(-alpha * y_binary * predictions)
                W /= np.sum(W)  # Normalize weights

                estimators_cls.append(stump)
                alphas_cls.append(alpha)

            # Store the estimators and alphas for the current class
            self.estimators.append(estimators_cls)
            self.alphas.append(alphas_cls)

    # Make predictions using the trained AdaBoost classifier for multi-class classification
    def predict(self, X):
        class_votes = np.zeros((X.shape[0], len(self.classes_)))

        for cls_idx, cls in enumerate(self.classes_):
            for estimator, alpha in zip(self.estimators[cls_idx], self.alphas[cls_idx]):
                predictions = estimator.predict(X)
                class_votes[:, cls_idx] += alpha * predictions

        if self.is_one_hot_encoded:
            # Return one-hot encoded predictions
            return (class_votes == class_votes.max(axis=1)[:, None]).astype(int)
        else:
            return self.classes_[np.argmax(class_votes, axis=1)]

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
        }

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
