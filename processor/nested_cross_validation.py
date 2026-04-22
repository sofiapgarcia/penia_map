"""
Functions for performing cross-validation for model selection and hyperparameter tuning.

Available Functions
-------------------
[Public]
nested_cross_val(...): Performs nested cross-validation using GroupKFold to evaluate a machine learning model with hyperparameter tuning.
tune_production_model(...): Performs hyperparameter tuning using GroupKFold, X and y.
------------------
[Private]
None
------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import pandas as pd
import numpy as np
from sklearn.base import RegressorMixin
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import r2_score
from typing import Dict, Any, List, Union

from constants import RANDOM_SEED


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #
def nested_cross_val(X: pd.DataFrame, y: pd.Series, estimator: RegressorMixin,
                     param_grid: Union[List[Dict[str, Any]], Dict[str, Any]],
                     splits_outer: int = 5, splits_inner: int = 2) -> pd.DataFrame():
    """
    Performs nested cross-validation using KFold to evaluate a machine learning regression model with hyperparameter tuning.

    The function applies a nested cross-validation strategy, where the outer loop performs model evaluation
    and the inner loop performs hyperparameter tuning using GridSearchCV.

    The function generates a pandas.DataFrame containing the hyperparameters of the model as well as the avg. R² score
    of the inner GridSearch, the R² score per outer fold, and the mean R² score and std over all folds.

    :param X: pandas.DataFrame containing the feature matrix
    :param y: pandas.Series containing the target values
    :param estimator: the estimator to be evaluated
    :param param_grid: the parameter grid for the estimator
    :param splits_outer: number of outer cross-validation splits. Default: 5
    :param splits_inner: number of inner cross-validation splits. Default: 2
    :return: pandas.DataFrame containing the best hyperparameters found for each outer fold,
        inner cross-validation R² score, outer fold R² score, and the overall mean and standard deviation
        of R² score across all outer folds.
    """

    # lists for holding the scores of the outer folds and the info of each fold
    fold_scores = []
    fold_info = []

    # set up cross-validation for outer loop
    outer_cv = KFold(n_splits=splits_outer, shuffle=True, random_state=RANDOM_SEED)

    # set up inner cross-validation for hyperparameter tuning using gridSearch
    inner_cv = KFold(n_splits=splits_inner, shuffle=True, random_state=RANDOM_SEED)
    grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring='r2',
                               n_jobs=-1, cv=inner_cv, verbose=1, refit=True)

    # run outer loop
    for fold_num, (train_idx, valid_idx) in enumerate(outer_cv.split(X, y), start=1):

        # get the corresponding X, y
        X_train = X.iloc[train_idx]
        X_val = X.iloc[valid_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[valid_idx]

        # run inner loop for hyperparameter optimization
        grid_search.fit(X_train, y_train)

        # test the best estimator on the validation set of the current fold (this is possible as refit=True)
        y_pred = grid_search.best_estimator_.predict(X_val)
        fold_r2 = r2_score(y_val, y_pred)

        # print the results
        inner_r2 = grid_search.best_score_
        best_params = grid_search.best_params_

        # print the results
        print(f'\nResults of fold {fold_num}/{splits_outer}')
        print(f'best R² score (inner fold avg.): {inner_r2:.4f}')
        print(f'best params: {best_params}')
        print(f'--> fold R² score: {fold_r2:.4f}')
        print('-------------------------')

        # append fold score
        fold_scores.append(fold_r2)

        # append the info for the fold
        best_params.update({'inner_r2': inner_r2, 'fold_r2': fold_r2})
        fold_info.append(best_params)

    # calculate the average R² score over all outer folds
    outer_fold_avg_r2 = np.mean(fold_scores)
    outer_fold_std_r2 = np.std(fold_scores)

    # print the result
    print(f'\naverage R² score over all folds: {outer_fold_avg_r2:.4f} +/- {outer_fold_std_r2:.4f}')
    print('### ----------------------------------------- ###')

    # generate a pandas DataFrame containing the info
    info_df = pd.DataFrame(fold_info)

    # add the average score
    # (adding here to not have it to re-calculate again. The entire column thus has the same value)
    info_df['estimator_avg_r2'] = outer_fold_avg_r2
    info_df['estimator_std_r2'] = outer_fold_std_r2

    return info_df


def tune_production_model(X: pd.DataFrame, y: pd.Series, estimator: RegressorMixin,
                          param_grid: Union[List[Dict[str, Any]], Dict[str, Any]],
                          cv_splits: int = 5):
    """
    Performs hyperparameter tuning using KFold and X, y. This function can be used to train the production model
    on the entire training set.
    :param X: pandas.DataFrame containing the feature matrix
    :param y: pandas.Series containing the target values
    :param estimator: the estimator to be evaluated
    :param param_grid: the parameter grid for the estimator
    :param cv_splits: the number of cross-validation splits for the gridsearch. Default: 5
    :return: the best estimator trained on the entire dataset with optimal hyperparameters
    """

    # set up cross-validation for hyperparameter tuning
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_SEED)
    grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring='r2',
                               n_jobs=-1, cv=cv, verbose=1, refit=True)

    # perform hyperparameter tuning
    grid_search.fit(X, y)

    # print the results
    print("\nResults of hyperparameter tuning")
    print(f"best R² score (CV avg.): {grid_search.best_score_:.4f}")
    print(f"best parameters: {grid_search.best_params_}")
    print('-------------------------')

    return grid_search.best_estimator_


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #