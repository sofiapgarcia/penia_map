"""
Functions to perform regression model development, including data splitting,
model selection using nested cross-validation, and final production model evaluation.

This module provides utilities to:
- Split a dataset into training and testing sets
- Perform model selection across multiple regression algorithms (Random Forest, KNN, SVM)
  with different normalization strategies
- Evaluate the final production model using standard regression metrics (R², MAE, RMSE)

Available Functions
-------------------
[Public]
run_training_pipeline(...): Splits the dataset and performs model selection on the training data.
evaluate_production_model(...): Tunes and evaluates the final regression model on training and test sets.

[Private]
_train_test_split(...): Splits the dataset into features/target and then into train/test sets.
_perform_model_selection(...): Iterates over normalization methods and evaluates models.
_evaluate_models(...): Performs nested cross-validation for multiple regression models.
_save_results(...): Saves evaluation results to disk.
"""
# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #

import pandas as pd
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from typing import List
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
import os
import numpy as np

# internal imports
from constants import RANDOM_SEED
from nested_cross_validation import nested_cross_val , tune_production_model
# ------------------------------------------------------------------------------------------------------------------- #
# file specific constants
# ------------------------------------------------------------------------------------------------------------------- #
RF = "Random Forest"
SVM = "SVM"
KNN = "KNN"
LR="LR"

SCALER = 'scaler'
PARAM_GRID = 'param_grid'
ESTIMATOR = 'estimator'


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #
def run_training_pipeline(df: pd.DataFrame,
                          target_column: str = "Estimated metabolic age",
                          test_size: float = 0.2,
                          random_state: int = 42,
                          normalization_methods: List[str] = None,
                          splits_outer: int = 5,
                          splits_inner: int = 2,
                          models_to_run: List[str] = None):
    """
    Runs the full training pipeline: splits the dataset and performs model selection.

    :param df: pandas.DataFrame containing the full dataset (features + target)
    :param target_column: name of the target column
    :param test_size: proportion of the dataset to include in the test split
    :param random_state: random seed for reproducibility
    :param normalization_methods: list of normalization methods to evaluate
    :param splits_outer: number of outer cross-validation splits. Default: 5
    :param splits_inner: number of inner cross-validation splits. Default: 2
    :param models_to_run: list of models to evaluate. Options: ['KNN', 'SVM', 'RF', 'LR'].
                          Default: None (runs all models)
    :return: X_train, X_test, y_train, y_test
    """

    # split dataset into train and test sets
    X_train, X_test, y_train, y_test = _split_data(
        df=df,
        target_column=target_column,
        test_size=test_size,
        random_state=random_state
    )

    print("\nDataset successfully split.")
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape:  {X_test.shape}\n")

    # perform model selection ONLY on training data
    # (important to avoid data leakage)
    _perform_model_selection(
        X_train=X_train,
        y_train=y_train,
        splits_outer=splits_outer,
        splits_inner=splits_inner,
        normalization_methods=normalization_methods,
        models_to_run=models_to_run
    )

    # return split data for later use (e.g., final evaluation)
    return X_train, X_test, y_train, y_test


def evaluate_production_model(X_train: pd.DataFrame, y_train: pd.Series,
                              X_test: pd.DataFrame, y_test: pd.Series,
                              cv_splits: int = 5) -> None:
    """
    Performs a final hyperparameter tuning on the production regression model and evaluates it on the test data.
    Prints common regression metrics such as R², MAE and RMSE for both training and test sets.

    :param X_train: pandas.DataFrame containing the training data
    :param y_train: pandas.Series containing the training targets
    :param X_test: pandas.DataFrame containing the test data
    :param y_test: pandas.Series containing the test targets
    :param cv_splits: the number of cross-validation splits for the grid search. Default: 5
    :return: None
    """

    # define regression estimator and hyperparameters
    estimator = RandomForestRegressor(random_state=RANDOM_SEED)

    param_grid = {
        "n_estimators": [50, 100, 500],
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5, 10]
    }

    # perform hyperparameter tuning (using R² as scoring)
    model = tune_production_model(
        X=X_train,
        y=y_train,
        estimator=estimator,
        param_grid=param_grid,
        cv_splits=cv_splits
    )

    # generate predictions for train and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # compute metrics for training set
    r2_train = r2_score(y_train, y_train_pred)  # coefficient of determination
    mae_train = mean_absolute_error(y_train, y_train_pred)  # mean absolute error
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))  # root mean squared error

    # compute metrics for test set
    r2_test = r2_score(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # print results
    print("\nResults on production model evaluation (Regression):")

    print("\nTrain set:")
    print(f"R2 score: {r2_train:.4f}")
    print(f"MAE:      {mae_train:.4f}")
    print(f"RMSE:     {rmse_train:.4f}")

    print("\nTest set:")
    print(f"R2 score: {r2_test:.4f}")
    print(f"MAE:      {mae_test:.4f}")
    print(f"RMSE:     {rmse_test:.4f}")


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #
def _split_data(df, target_column="Estimated metabolic age", test_size=0.2, random_state=42):
    """
    Split a DataFrame into features and target, and then into train and test sets.

    :param df: Input DataFrame
    :param target_column: Name of the target column
    :param test_size: Proportion of the dataset to include in the test split
    :param random_state: Random seed for reproducibility
    :return: X_train, X_test, y_train, y_test
    """

    # Separate features (X) and target (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

def _perform_model_selection(X_train: pd.DataFrame, y_train: pd.Series, splits_outer: int, splits_inner: int,
                            normalization_methods: List[str] = None, models_to_run: List[str] = None) -> None:
    """
    Evaluates multiple regression models (Random Forest, KNN, SVM, and Linear Regression) using nested cross-validation
    to select which model is best for production. The training data should already be divided from test data.

    :param X_train: pandas.DataFrame containing the training features
    :param y_train: pandas.Series containing the training target values
    :param normalization_methods: list of normalization methods to evaluate.
                                  Options: 'standard', 'minmax', 'robust', 'none'.
                                  Default: ['standard', 'minmax', 'robust', 'none']
    :param models_to_run: list of models to evaluate. Options: ['KNN', 'SVM', 'RF', 'LR'].
                          Default: None (runs all models)
    :param splits_outer: number of outer cross-validation splits.
    :param splits_inner: number of inner cross-validation splits.
    :return: None
    """

    if normalization_methods is None:
        normalization_methods = ['standard', 'minmax', 'robust', 'none']

    print(f"Total number of instances for training: {X_train.shape[0]}")
    print(f"Number of features: {X_train.shape[1]}\n")

    # Evaluate each normalization method
    for norm_method in normalization_methods:
        print("\n" + "=" * 60)
        print(f"EVALUATING WITH NORMALIZATION: {norm_method.upper()}")
        print("=" * 60 + "\n")

        # evaluate the models
        _evaluate_models(
            X_train,
            y_train,
            normalization_method=norm_method,
            splits_inner=splits_inner,
            splits_outer=splits_outer,
            models_to_run=models_to_run
        )


def _evaluate_models(X_train: pd.DataFrame, y_train: pd.Series, splits_outer: int, splits_inner: int,
                     normalization_method: str = 'standard', models_to_run: List[str] = None) -> None:
    """
    Evaluates multiple machine learning models using nested cross-validation with a specified normalization method.

    This function initializes and evaluates different regression models (SVR, KNN, Random Forest, and Linear Regression)
    with the specified normalization technique applied within the pipeline. It applies nested
    cross-validation to assess model performance and saves the results.

    :param X_train: pandas.DataFrame containing the training data
    :param y_train: pandas.Series containing the target values
    :param normalization_method: normalization method to apply in the pipeline. Options: 'standard', 'minmax', 'robust', 'none'.
                                 Default: 'standard'
    :param splits_outer: number of outer cross-validation splits.
    :param splits_inner: number of inner cross-validation splits.
    :param models_to_run: list of models to evaluate. Options: ['KNN', 'SVM', 'RF', 'LR'].
                          Default: None (runs all models)
    :return: None
    """

    # Default models
    if models_to_run is None:
        models_to_run = [KNN, SVM, RF, LR]

    # Dictionary mapping normalization method names to scaler objects
    scaler_dict = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler(),
        'none': None
    }

    if normalization_method not in scaler_dict:
        raise ValueError(f"Invalid normalization method. Choose from: {list(scaler_dict.keys())}")

    scaler = scaler_dict[normalization_method]

    # Build pipelines with the appropriate normalization
    if normalization_method == 'none':
        knn_pipeline = Pipeline([(KNN, KNeighborsRegressor(algorithm='ball_tree'))])
        svm_pipeline = Pipeline([(SVM, SVR())])
        lr_pipeline = Pipeline([(LR, linear_model.LinearRegression())])
    else:
        knn_pipeline = Pipeline([(SCALER, scaler), (KNN, KNeighborsRegressor(algorithm='ball_tree'))])
        svm_pipeline = Pipeline([(SCALER, scaler), (SVM, SVR())])
        lr_pipeline = Pipeline([(SCALER, scaler), (LR, linear_model.LinearRegression())])

    model_dict = {

         KNN: {ESTIMATOR: knn_pipeline,
                 PARAM_GRID: [{f'{KNN}__n_neighbors': list(range(1, 15)), f'{KNN}__p': [1, 2]}]},

         SVM: {ESTIMATOR: svm_pipeline, PARAM_GRID: [
             {f'{SVM}__kernel': ['rbf'], f'{SVM}__C': np.power(10., np.arange(-4, 4)),
              f'{SVM}__gamma': np.power(10., np.arange(-5, 0))},
             {f'{SVM}__kernel': ['linear'], f'{SVM}__C': np.power(10., np.arange(-4, 4))}]},

         RF: {ESTIMATOR: RandomForestRegressor(random_state=RANDOM_SEED), PARAM_GRID: [
             {"criterion": ['squared_error', 'absolute_error'], "n_estimators": [50, 100, 500, 1000],
              "max_depth": [2, 5, 10, 20, 30]}]},

         LR: {ESTIMATOR: lr_pipeline, PARAM_GRID: [{}]}  # no hyperparameters
    }

    # Filter models
    selected_models = {k: v for k, v in model_dict.items() if k in models_to_run}

    # Evaluate each model
    for model_name, param_dict in selected_models.items():
        print('\n### ----------------------------------------- ###')
        print(f'Algorithm: {model_name}')

        est = param_dict[ESTIMATOR]
        param_grid_est = param_dict[PARAM_GRID]

        info_df = nested_cross_val(X_train, y_train, estimator=est, param_grid=param_grid_est,
                                   splits_outer=splits_outer, splits_inner=splits_inner)

        _save_results(info_df, estimator_name=model_name, normalization_method=normalization_method)

def _save_results(info_df: pd.DataFrame, estimator_name: str, normalization_method: str) -> None:
    """
    Saves the results from the nested cross-validation into a .csv file.

    :param info_df: pandas.DataFrame containing the information from the nested cross-validation
    :param estimator_name: the name of the estimator used in the nested cross-validation
    :param normalization_method: the normalization method used
    :return: None
    """

    # get the path to the current project
    project_path = os.getcwd()

    # create results directory (if it doesn't exist)
    results_dir = os.path.join(project_path, "Results", "ML")
    os.makedirs(results_dir, exist_ok=True)

    # create full file path
    file_path = os.path.join(results_dir,
                             f'{estimator_name}_norm-{normalization_method}.csv')

    info_df.to_csv(file_path, sep=';', index=False)
    print(f'Results saved to: {file_path}')