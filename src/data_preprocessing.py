"""
Implemention of the Data Preprocessing module. This module is responsible for performing the data preprocessing on the dataset.

Author
------
@rkalai
"""

# 3rd party libraries
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    LabelEncoder,
    OrdinalEncoder,
    OneHotEncoder,
)

# Import the libraries for the UCIML Repository
from ucimlrepo import fetch_ucirepo

# Import the necessary libraries
# Custom libraries
import src.constants as CONSTANTS

# GLOBAL SETTINGS
# Set Seed for reproducibility
np.random.seed(CONSTANTS.RANDOM_STATE)


def select_uci_dataset(
    dataset_id: str = None,
    categorical_encoding_scheme: str = "label",
    impute_missing_values_method: str = "mean",
    drop_missing_values: bool = True,
    scale_data: bool = True,
    outlier_treatment_method=None,
):
    """
    Select the dataset from the UCIML Repository based on the dataset ID.

    Parameters
    ----------
    dataset_id (str): The ID of the dataset to be selected.
    categorical_encoding_scheme (str): The encoding scheme for the categorical variables. Default is 'label', but can also be 'one-hot' or 'ordinal'.
    impute_missing_values_method (str): The method to impute the missing values. Default is 'mean', but can also be 'median' or 'mode'.
    drop_missing_values (bool): Whether to drop the missing values or impute them.
    scale_data (bool): Whether to scale the data or not.
    outlier_treatment_method (str): The method to treat the outliers. Default is None, but can also be 'iqr' or 'z-score'.

    Returns
    -------
    pd.DataFrame: The dataset from the UCIML Repository retrieved based on the dataset ID.
    """
    uci_dataset = fetch_ucirepo(id=dataset_id)
    # Create a Mapping of the type of encoding that is to be performed for each Dataset
    features = uci_dataset.data.features.columns
    targets = uci_dataset.data.targets.columns
    # Drop any columns that are not features or targets
    modelling_data = uci_dataset.data.original[list(features) + list(targets)]
    dataset_name = uci_dataset.metadata.name
    # Preprocess the data
    modelling_data = data_preprocessing(
        data=modelling_data,
        drop_missing_values=drop_missing_values,
        impute_missing_values_method=impute_missing_values_method,
        encode_categorical_variables=categorical_encoding_scheme,
        scale_data=scale_data,
        outlier_treatment_method=outlier_treatment_method,
    )
    # Feature Headers, Target Headers
    feature_headers = [
        header
        for header in modelling_data.columns
        if any(feature_name in header for feature_name in features)
    ]
    target_headers = [
        header
        for header in modelling_data.columns
        if any(target_name in header for target_name in targets)
    ]
    # Assuming `data` is your DataFrame after preprocessing
    X = modelling_data[feature_headers].reset_index(drop=True)
    y = modelling_data[target_headers].reset_index(drop=True)
    # Convert to numpy arrays if they're not already
    X = X.to_numpy()
    y = y.to_numpy()
    # If either of the dimensions is 1, then convert it to a 1D array
    if X.shape[1] == 1:
        X = X.ravel()
    if y.shape[1] == 1:
        y = y.ravel()
    return X, y, feature_headers, target_headers, dataset_name


def data_preprocessing(
    data: pd.DataFrame = None,
    drop_missing_values: bool = False,
    impute_missing_values_method: str = "mean",
    encode_categorical_variables: str = "label",
    scale_data: bool = False,
    outlier_treatment_method=None,
):
    """
    Perform data preprocessing on the dataset. This includes:
    1. Missing Value Treatment: Imputation with the mean/mode/median of the column or drop the missing values.
    2. Data Transformation: Encode the categorical variables to numerical variables using Label Encoding or One-Hot Encoding. Default is Label Encoding.
    3. Data Scaling: Scale the data using Standardization or Min-Max Scaling.
    4. Outlier Treatment: Treat the outliers using the specified method, if None then no treatment is performed.

    Parameters
    ----------
    data (pd.DataFrame): The dataset to be preprocessed.
    drop_missing_values (bool): Whether to drop the missing values or impute them.
    impute_missing_values_method (str): The method to impute the missing values. Default is 'mean', but can also be 'median' or 'mode'.
    encode_categorical_variables (str): The method to encode the categorical variables. Default is 'label', but can also be 'one-hot' or 'ordinal'. If None, then no encoding is performed.
    scale_data (bool): Whether to scale the data or not.
    outlier_treatment_method (str): The method to treat the outliers. Default is None, but can also be 'iqr' or 'z-score'.

    Returns
    -------
    pd.DataFrame: The preprocessed dataset.
    """
    # Segregate the data into Categories and Numerical variables
    # For Categories, impute the missing values with the mode, regardless of the impute_missing_values_method
    # For Numerical variables, impute the missing values with the mean/median, based on the impute_missing_values_method
    if drop_missing_values:
        data = data.dropna(how="any", axis=0)
        data = data.reset_index(drop=True)
    categorical_data = data.select_dtypes(include="object")
    categorical_columns = categorical_data.columns
    numerical_data = data.select_dtypes(include="number")
    if not numerical_data.empty:
        if impute_missing_values_method == "mean":
            imputer = SimpleImputer(strategy="mean")
        elif impute_missing_values_method == "median":
            imputer = SimpleImputer(strategy="median")
        elif impute_missing_values_method == "mode":
            imputer = SimpleImputer(strategy="most_frequent")
        else:
            raise ValueError(
                'Invalid imputation method. Please specify either "mean", "median" or "mode".'
            )
        # Impute the values in the numerical data, and return a DataFrame with the imputed values
        numerical_data = pd.DataFrame(
            imputer.fit_transform(numerical_data), columns=numerical_data.columns
        )
        # Treat the outliers in the dataset using the specified method
        if outlier_treatment_method:
            numerical_data = outlier_treatment(
                numerical_data, method=outlier_treatment_method
            )
        if scale_data:
            numerical_data = perform_data_scaling(numerical_data)
    if encode_categorical_variables and not categorical_data.empty:
        categorical_data = categorical_data.apply(lambda x: x.fillna(x.mode().iloc[0]))
        # Encode the categorical variables to numerical variables using Label Encoding or One-Hot Encoding
        # Label Encoding: Convert the categories to numerical values - Set as Default
        # One-Hot Encoding: Create dummy variables for each category
        if encode_categorical_variables == "label":
            encoder = LabelEncoder()
            encoded_data = categorical_data.apply(encoder.fit_transform)
            encoded_data_columns = categorical_columns
        elif encode_categorical_variables == "ordinal":
            encoder = OrdinalEncoder()
            encoded_data = encoder.fit_transform(categorical_data)
            encoded_data_columns = categorical_columns
        elif encode_categorical_variables == "one-hot":
            encoder = OneHotEncoder(sparse_output=False)
            encoded_data = encoder.fit_transform(categorical_data)
            encoded_data_columns = encoder.get_feature_names_out(categorical_columns)
        else:
            raise ValueError(
                'Invalid encoding method. Please specify either "label" or "one-hot".'
            )
        # Convert the encoded data into a DataFrame with appropriate column names
        encoded_df = pd.DataFrame(data=encoded_data, columns=encoded_data_columns)
        # Convert all the columns of the Encoded DataFrame to int type
        encoded_df = encoded_df.astype(int)
        # Concatenate the original numerical data with the new one-hot encoded data
        data = pd.concat([numerical_data, encoded_df], axis=1).reset_index(drop=True)
    return data


def perform_data_scaling(data: pd.DataFrame = None, scaling_method: str = "min-max"):
    """
    Scale the dataset using the specified scaling method.

    Parameters
    ----------
    data (pd.DataFrame): The dataset to be scaled.
    scaling_method (str): The method to scale the dataset. Default is 'standard'.

    Returns
    -------
    pd.DataFrame: The scaled dataset.
    """
    if scaling_method == "standard":
        scaler = StandardScaler()
    elif scaling_method == "min-max":
        scaler = MinMaxScaler()
    else:
        raise ValueError(
            'Invalid scaling method. Please specify either "standard" or "min-max".'
        )
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    return data


def outlier_treatment(data: pd.DataFrame = None, method: str = "iqr"):
    """
    Treat the outliers in the dataset using the specified method.

    Parameters
    ----------
    data (pd.DataFrame): The dataset to be treated for outliers.
    method (str): The method to treat the outliers. Default is 'iqr', but can also be 'z-score'.

    Returns
    -------
    pd.DataFrame: The dataset with the treated outliers.
    """
    if method == "iqr":
        # The way to treat the outliers is by removing them from the dataset
        # if the value is less than Q1 - 1.5 * IQR or greater than Q3 + 1.5 * IQR
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        data = data[
            ~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)
        ]
    elif method == "z-score":
        z = np.abs(stats.zscore(data))
        data = data[(z < 3).all(axis=1)]
    else:
        raise ValueError('Invalid outlier treatment method. Please specify "iqr".')
    data.reset_index(drop=True, inplace=True)
    return data
