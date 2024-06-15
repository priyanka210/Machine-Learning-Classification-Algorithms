"""
Implementation of K-Nearest Neighbors (KNN) algorithm from scratch.

Authors
-------
@schettiar, sadhana.chettiar@unb.ca
@rkalai, rishabh.kalai@unb.ca
"""

import heapq

# Import the necessary libraries
# 3rd party libraries
import numpy as np

# Custom libraries
import src.utilities as UTILITIES


class KNN:
    def __init__(self, k="auto", distance_fn=UTILITIES.euclidean_distance):
        self.k = k
        self.distance_fn = distance_fn
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        # Determine k as the cube root of the number of training samples
        if self.k == "auto":
            self.k = int(np.cbrt(len(X_train)))
        return self

    def _predict_single_point(self, x):
        distances = [
            (self.distance_fn(x, x_train), y)
            for x_train, y in zip(self.X_train, self.y_train)
        ]
        k_nearest_neighbors = heapq.nsmallest(self.k, distances, key=lambda x: x[0])
        k_nearest_labels = [neighbor[1] for neighbor in k_nearest_neighbors]
        # The data is one hot encoded, so finding the most common label is a matter of summing the columns
        most_common_label = np.sum(k_nearest_labels, axis=0)
        # Find the index of the most common label
        most_common_label = np.argmax(most_common_label)
        return most_common_label

    def predict(self, X_test):
        predictions = np.array([self._predict_single_point(x) for x in X_test])
        return predictions

    def get_params(self, deep=True):
        return {"k": self.k, "distance_fn": self.distance_fn}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
