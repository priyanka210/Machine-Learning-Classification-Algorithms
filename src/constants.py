"""
File Path Configurations, Constants, and Global Variables for the Project
"""

import os

import numpy as np

# Project Root Directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# Model Directory
MODEL_DIR = os.path.join(ROOT_DIR, "models")

# Random State Seed
RANDOM_STATE = 42

# Dataset ID Constants
BREAST_CANCER_ID = 17
CAR_EVALUATION_ID = 19
ECOLI_ID = 39
LETTER_RECOGNITION_ID = 59
MUSHROOM_ID = 73

# Configuration Settings
# Data Preprocessing
DROP_MISSING_VALUES = True
IMPUTE_MISSING_VALUES_METHOD = "mean"
ONE_HOT_ENCODING = "one-hot"
ORDINAL_ENCODING = "ordinal"
SCALE_DATA = False
OUTLIER_TREATMENT_METHOD = None
# Train-Test Split
TEST_SIZE = 0.2
TRAIN_SIZE = 1 - TEST_SIZE
# Cross-Validation Settings
N_SPLITS = 5
N_REPEATS = 1

# Model Evaluation Metrics
# Artificial Neural Network
# ANN Default Hyperparameters
HIDDEN_SIZE = 100
LEARNING_RATE = 0.01
EPOCHS = 10000
# ANN Hyperparameters
ANN_HYPERPARAMETER_GRID = {
    "hidden_size": [50, 100, 150, 200],
    "learning_rate": [0.01, 0.1, 0.5, 1, 1.5],
    "epochs": [10000, 25000, 50000],
}
# Best Hyperparameters for ANN that were found using Grid Search
ANN_HYPERPARAMETERS = {
    ECOLI_ID: {"epochs": 500, "hidden_size": 150, "learning_rate": 1},
    BREAST_CANCER_ID: {"epochs": 500, "hidden_size": 50, "learning_rate": 1},
    LETTER_RECOGNITION_ID: {"epochs": 25000, "hidden_size": 150, "learning_rate": 1.5},
    MUSHROOM_ID: {"epochs": 500, "hidden_size": 50, "learning_rate": 1},
    CAR_EVALUATION_ID: {"epochs": 1000, "hidden_size": 50, "learning_rate": 1.5},
}
# Random Forest
# Random Forest Default Hyperparameters
RF_N_ESTIMATORS = 50
RF_MAX_DEPTH = 10
# Random Forest Hyperparameters
RANDOM_FOREST_HYPERPARAMETER_GRID = {
    "n_estimators": np.linspace(5, 60, 5).astype(int),
    "max_depth": np.linspace(5, 60, 5).astype(int),
}
# Best Hyperparameters for Random Forest that were found using Grid Search
RANDOM_FOREST_HYPERPARAMETERS = {
    CAR_EVALUATION_ID: {"max_depth": 15, "n_estimators": 25},
    ECOLI_ID: {"max_depth": 46, "n_estimators": 35},
    LETTER_RECOGNITION_ID: {"max_depth": 5, "n_estimators": 16},
    MUSHROOM_ID: {"max_depth": 5, "n_estimators": 10},
    BREAST_CANCER_ID: {"max_depth": 5, "n_estimators": 10},
}

# AdaBoost
# AdaBoost Default Hyperparameters
ADABOOST_N_ESTIMATORS = 100
ADABOOST_LEARNING_RATE = 0.1
# AdaBoost Hyperparameters
ADABOOST_HYPERPARAMETER_GRID = {
    "n_estimators": np.linspace(100, 5100, 200).astype(int),
    "learning_rate": [0.1, 0.25],
}
# Best Hyperparameters for AdaBoost that were found using Grid Search
ADABOOST_HYPERPARAMETERS = {
    BREAST_CANCER_ID: {"learning_rate": 0.25, "n_estimators": 100},
    CAR_EVALUATION_ID: {"learning_rate": 0.25, "n_estimators": 2500},
    ECOLI_ID: {"learning_rate": 0.25, "n_estimators": 100},
    LETTER_RECOGNITION_ID: {"learning_rate": 0.25, "n_estimators": 4000},
    MUSHROOM_ID: {"learning_rate": 0.25, "n_estimators": 300},
}
# Data Encoding Scheme
# Random Forest Data Encoding Scheme
RANDOM_FOREST_DATA_ENCODING_SCHEME = "label"
# Artificial Neural Network Data Encoding Scheme
ANN_DATA_ENCODING_SCHEME = "one-hot"
# AdaBoost Data Encoding Scheme
ADABOOST_DATA_ENCODING_SCHEME = "label"
# KNN Data Encoding Scheme
KNN_DATA_ENCODING_SCHEME = "one-hot"
