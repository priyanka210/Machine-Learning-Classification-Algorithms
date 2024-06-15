"""
Implementation of Random Forest module. This module is responsible for implementing the Random Forest algorithm from scratch.

Author
------
@rkalai, rishabh.kalai@unb.ca
"""

# Suppress All Warnings
import warnings

warnings.filterwarnings("ignore")
# Import the necessary libraries
# 3rd party libraries
import numpy as np
import random
from sklearn.metrics import accuracy_score
from math import sqrt
# Custom libraries
import src.utilities as UTILITIES
import src.constants as CONSTANTS


class Node:
    def __init__(
            self, feature_index=None, threshold=None, left=None, right=None, value=None
    ):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left  # Left subtree
        self.right = right  # Right subtree
        self.value = value  # Value at leaf node


class DecisionTree:
    def __init__(self, max_depth=None, max_features=None):
        self.root = None
        self.max_depth = max_depth
        self.max_features = max_features

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _gini_index(self, y):
        # Calculate the Gini Index for a node
        m = len(y)
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))

    def _information_gain(self, parent, l_child, r_child):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        gain = self._gini_index(parent) - (
                weight_l * self._gini_index(l_child) + weight_r * self._gini_index(r_child)
        )
        return gain

    def _get_best_split(self, X, y, num_features):
        best_split = {}
        max_gain = -float("inf")
        features = random.sample(range(num_features), self.max_features)
        for feature_index in features:
            feature_values = X[:, feature_index]
            thresholds = np.unique(feature_values)
            for threshold in thresholds:
                left_indices = np.where(feature_values <= threshold)
                right_indices = np.where(feature_values > threshold)
                gain = self._information_gain(y, y[left_indices], y[right_indices])
                if gain > max_gain:
                    best_split["feature_index"] = feature_index
                    best_split["threshold"] = threshold
                    best_split["dataset_left"] = left_indices
                    best_split["dataset_right"] = right_indices
                    best_split["gain"] = gain
                    max_gain = gain
        return best_split

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        # Criteria to stop splitting
        if num_samples >= 2 and depth <= self.max_depth:
            best_split = self._get_best_split(X, y, num_features)
            if best_split["gain"] > 0:
                left_subtree = self._build_tree(
                    X[best_split["dataset_left"][0]],
                    y[best_split["dataset_left"][0]],
                    depth + 1,
                )
                right_subtree = self._build_tree(
                    X[best_split["dataset_right"][0]],
                    y[best_split["dataset_right"][0]],
                    depth + 1,
                )
                return Node(
                    best_split["feature_index"],
                    best_split["threshold"],
                    left_subtree,
                    right_subtree,
                )
        leaf_value = self._calculate_leaf_value(y)
        return Node(value=leaf_value)

    def _calculate_leaf_value(self, y):
        # Calculate the most common target value in the segment
        leaf_value = max(y, key=list(y).count)
        return leaf_value

    def _traverse_tree(self, X, node):
        if node.value is not None:
            return node.value
        if X[node.feature_index] <= node.threshold:
            return self._traverse_tree(X, node.left)
        else:
            return self._traverse_tree(X, node.right)

    def predict(self, X):
        # Predict function to traverse the tree for each sample and return prediction
        return np.array([self._traverse_tree(x, self.root) for x in X])


class RandomForest:
    def __init__(self, n_estimators: int = CONSTANTS.RF_N_ESTIMATORS, max_depth: int = CONSTANTS.RF_MAX_DEPTH,
                 max_features: int = None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        num_features = X.shape[1]
        # Set max_features to sqrt(num_features) if not specified, this is because it is a common rule of thumb to use sqrt(num_features) for classification tasks
        self.max_features = (
            int(sqrt(num_features)) if not self.max_features else self.max_features
        )
        # Initialize the trees in the Random Forest
        self.trees = [
            DecisionTree(max_depth=self.max_depth, max_features=self.max_features)
            for _ in range(self.n_estimators)
        ]
        for tree in self.trees:
            X_sample, y_sample = UTILITIES.bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)

    def predict(self, X):
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        # Return the mode of the predictions
        return np.array(
            [
                np.bincount(tree_predictions[:, i]).argmax()
                for i in range(tree_predictions.shape[1])
            ]
        )

    def get_params(self, deep=True):
        # deep=True is set by default, as the Random Forest does not have any Sub-Models
        # It has no effect on the function
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "max_features": self.max_features,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)