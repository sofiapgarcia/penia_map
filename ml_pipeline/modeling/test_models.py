"""
Functions to perform regression model development with optimized results tracking.

Available Functions
-------------------
[Public]
run_training_pipeline(...): Runs model selection using nested cross-validation.
get_best_models_summary(...): Returns best model per algorithm.

[Private]
_init_results_tracker(...): Initializes results structure.
_perform_model_selection(...): Iterates over normalization methods.
_evaluate_models(...): Evaluates models.
_track_results(...): Stores results.
_extract_best_params(...): Extracts best parameters.
_build_pipeline(...): Builds pipeline with optional scaler.
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #

import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

# internal imports
from ml_pipeline.modeling.nested_cross_validation import nested_cross_val , tune_production_model
# ------------------------------------------------------------------------------------------------------------------- #
# file specific constants
# ------------------------------------------------------------------------------------------------------------------- #
RANDOM_SEED = 42
RF = "Random Forest"
SVM = "SVM"
KNN = "KNN"
LR = "LR"

SCALER = 'scaler'
PARAM_GRID = 'param_grid'
ESTIMATOR = 'estimator'

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def run_training_pipeline(
        train_df: pd.DataFrame,
        target_column: str = "Estimated metabolic age",
        normalization_methods=None,
        splits_outer: int = 5,
        splits_inner: int = 2,
        models_to_run=None
) -> Dict:
    """
    Runs the model selection pipeline using only training data.

    :param train_df: Training DataFrame (features + target)
    :param target_column: Name of the target column
    :param normalization_methods: List of normalization methods
    :param splits_outer: Number of outer CV splits
    :param splits_inner: Number of inner CV splits
    :param models_to_run: Models to evaluate
    :return: Dictionary with results
    """

    # Separate features (X) and target (y) from the full DataFrame
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    # Initialize results container
    results = {
        'all_results': []
    }

    # Perform model selection using nested cross-validation
    results = _perform_model_selection(
        X_train=X_train,
        y_train=y_train,
        splits_outer=splits_outer,
        splits_inner=splits_inner,
        normalization_methods=normalization_methods,
        models_to_run=models_to_run,
        results=results
    )

    # Generate summary of best models
    results_summary = _get_best_models_summary(results)

    return results_summary

def evaluate_production_model_simple(train_df: pd.DataFrame,
                              test_df: pd.DataFrame,
                              target_column: str = "Estimated metabolic age",
                              cv_splits: int = 5) -> None:
    """
    Performs a final hyperparameter tuning on the production regression model and evaluates it on the test data.
    Prints common regression metrics such as R², MAE and RMSE for both training and test sets.

    :param train_df: pandas.DataFrame containing the training data (features + target)
    :param test_df: pandas.DataFrame containing the test data (features + target)
    :param target_column: name of the target column
    :param cv_splits: the number of cross-validation splits for the grid search. Default: 5
    :return: None
    """

    # Separate features (X) and target (y) from training data
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    # Separate features (X) and target (y) from test data
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    # Define regression estimator and hyperparameters
    estimator = RandomForestRegressor(random_state=RANDOM_SEED)

    param_grid = {
        "n_estimators": [50, 75, 100, 200, 500],
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_leaf": [1, 3, 5, 10],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", "log2", None]
    }

    # Perform hyperparameter tuning on training data only (to avoid data leakage)
    model = tune_production_model(
        X=X_train,
        y=y_train,
        estimator=estimator,
        param_grid=param_grid,
        cv_splits=cv_splits
    )

    # Generate predictions for both training and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Compute metrics for training set
    r2_train = r2_score(y_train, y_train_pred)  # coefficient of determination
    mae_train = mean_absolute_error(y_train, y_train_pred)  # mean absolute error
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))  # root mean squared error

    # Compute metrics for test set (unseen data)
    r2_test = r2_score(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # Print results
    print("\nResults on production model evaluation (Regression):")

    print("\nTrain set:")
    print(f"R2 score: {r2_train:.4f}")
    print(f"MAE:      {mae_train:.4f}")
    print(f"RMSE:     {rmse_train:.4f}")

    print("\nTest set:")
    print(f"R2 score: {r2_test:.4f}")
    print(f"MAE:      {mae_test:.4f}")
    print(f"RMSE:     {rmse_test:.4f}")


def evaluate_production_model_k_fold(df: pd.DataFrame,
                               target_column: str = "Estimated metabolic age",
                               n_splits: int = 5,
                               param_grid: Dict = None,
                               inner_cv_splits: int = 5,
                               n_strat_bins: int = 5,
                               plot_distribution: bool = True) -> pd.DataFrame:
    """
    Evaluates the production model using k-fold cross-validation over the full
    dataset, with folds stratified by the target variable (binned into
    quantile-based groups) to ensure a similar target distribution across
    folds. Hyperparameter tuning is performed independently within each fold's
    training data only, to avoid data leakage into the validation fold.

    :param df: full DataFrame (features + target), e.g. df_reduced
    :param target_column: name of the target column
    :param n_splits: number of outer k-fold splits (used for evaluation)
    :param param_grid: hyperparameter grid for the Random Forest
    :param inner_cv_splits: number of CV splits used for hyperparameter tuning
        within each outer fold's training data
    :param n_strat_bins: number of quantile bins used to stratify the
        continuous target for fold splitting
    :param plot_distribution: whether to plot the target distribution per fold
    :return: DataFrame containing the metrics and best hyperparameters for each fold
    """

    X = df.drop(columns=[target_column])
    y = df[target_column]

    if param_grid is None:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7, None],
            "min_samples_leaf": [3, 5, 10],
        }

    y_binned = pd.qcut(y, q=n_strat_bins, labels=False, duplicates="drop")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    fold_results = []
    fold_target_data = []  # holds target values per fold, for plotting

    for fold_num, (train_idx, val_idx) in enumerate(skf.split(X, y_binned), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Store target values for the distribution plot
        fold_target_data.append({
            "fold": fold_num,
            "train_values": y_train.values,
            "val_values": y_val.values,
        })

        estimator = RandomForestRegressor(random_state=RANDOM_SEED)
        best_model = tune_production_model(
            X=X_train, y=y_train, estimator=estimator,
            param_grid=param_grid, cv_splits=inner_cv_splits
        )

        y_train_pred = best_model.predict(X_train)
        y_val_pred = best_model.predict(X_val)

        r2_train = r2_score(y_train, y_train_pred)
        mae_train = mean_absolute_error(y_train, y_train_pred)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

        r2_val = r2_score(y_val, y_val_pred)
        mae_val = mean_absolute_error(y_val, y_val_pred)
        rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))

        print(f"\n--- Fold {fold_num}/{n_splits} ---")
        print(f"Train target range: [{y_train.min():.1f}, {y_train.max():.1f}] "
              f"(mean={y_train.mean():.1f})")
        print(f"Val target range:   [{y_val.min():.1f}, {y_val.max():.1f}] "
              f"(mean={y_val.mean():.1f})")
        print(f"Best params: n_estimators={best_model.get_params()['n_estimators']}, "
              f"max_depth={best_model.get_params()['max_depth']}, "
              f"min_samples_leaf={best_model.get_params()['min_samples_leaf']}")
        print(f"Train  -> R²: {r2_train:.4f} | MAE: {mae_train:.4f} | RMSE: {rmse_train:.4f}")
        print(f"Val    -> R²: {r2_val:.4f} | MAE: {mae_val:.4f} | RMSE: {rmse_val:.4f}")

        fold_results.append({
            "fold": fold_num,
            "n_estimators": best_model.get_params()["n_estimators"],
            "max_depth": best_model.get_params()["max_depth"],
            "min_samples_leaf": best_model.get_params()["min_samples_leaf"],
            "r2_train": r2_train, "mae_train": mae_train, "rmse_train": rmse_train,
            "r2_val": r2_val, "mae_val": mae_val, "rmse_val": rmse_val,
        })

    results_df = pd.DataFrame(fold_results)

    print("\n=== Average k-fold results (validation) ===")
    print(f"R²:   {results_df['r2_val'].mean():.4f} +/- {results_df['r2_val'].std():.4f}")
    print(f"MAE:  {results_df['mae_val'].mean():.4f} +/- {results_df['mae_val'].std():.4f}")
    print(f"RMSE: {results_df['rmse_val'].mean():.4f} +/- {results_df['rmse_val'].std():.4f}")

    # Plot target distribution across folds to visually confirm stratification
    if plot_distribution:
        _plot_target_distribution_by_fold(fold_target_data, target_column)

    return results_df

def split_train_test_df(
        df: pd.DataFrame,
        target_col: str = "Estimated metabolic age",
        test_size: float = 0.2,
        random_state: int = RANDOM_SEED,
        n_bins: int = 5
) -> tuple:
    """
    Splits a DataFrame into training and testing sets, preserving the
    distribution of the target variable (metabolic age) between the two sets.

    :param df: Input DataFrame
    :param target_col: Name of the target column (metabolic age) used for stratification
    :param test_size: Proportion of the dataset for testing
    :param random_state: Random seed for reproducibility
    :param n_bins: Number of bins used to stratify the split
    :return: train_df, test_df
    """
    target_bins = pd.qcut(df[target_col], q=n_bins, duplicates="drop")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=target_bins
    )

    # Reset indices to avoid misalignment issues later
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Print dataset sizes for verification
    print("\nDataset split summary:")
    print(f"Total rows: {len(df)}")
    print(f"Train set: {len(train_df)} rows ({len(train_df) / len(df):.2%})")
    print(f"Test set:  {len(test_df)} rows ({len(test_df) / len(df):.2%})")

    print("\nTarget distribution check:")
    print(f"Train - mean: {train_df[target_col].mean():.2f}, std: {train_df[target_col].std():.2f}")
    print(f"Test  - mean: {test_df[target_col].mean():.2f}, std: {test_df[target_col].std():.2f}")

    return train_df, test_df


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _get_best_models_summary(results: Dict) -> pd.DataFrame:
    """
    Returns best model per algorithm.

    :param results: Results dictionary from pipeline
    :return: DataFrame with best models
    """

    # Handle empty results
    if not results['all_results']:
        print("️ No results available.")
        return pd.DataFrame()

    summary_data = []

    # Convert results to tabular format
    for result in results['all_results']:
        summary_data.append({
            'Algorithm': result['model_name'],
            'Normalization': result['normalization_method'],
            'R² Mean': result['estimator_avg_r2'],
            'R² Std': result['estimator_std_r2'],
            'MAE Mean': result['estimator_avg_mae'],
            'MAE Std': result['estimator_std_mae'],
            'Best Hyperparameters': result['best_params_summary']
        })

    summary_df = pd.DataFrame(summary_data)

    # Sort by performance
    summary_df = summary_df.sort_values('R² Mean', ascending=False).reset_index(drop=True)

    # Select best per algorithm
    best_per_algorithm = summary_df.loc[
        summary_df.groupby('Algorithm')['R² Mean'].idxmax()
    ]

    return best_per_algorithm



def _perform_model_selection(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        splits_outer: int,
        splits_inner: int,
        normalization_methods: List[str],
        models_to_run: List[str],
        results: Dict
) -> Dict:
    """
    Iterates over normalization methods and evaluates models.

    :param X_train: Training features
    :param y_train: Training target
    :param splits_outer: Outer CV splits
    :param splits_inner: Inner CV splits
    :param normalization_methods: Normalization methods
    :param models_to_run: Models to evaluate
    :param results: Results dictionary
    :return: Updated results dictionary
    """

    # If no normalization methods are provided, evaluate all supported options
    if normalization_methods is None:
        normalization_methods = ['standard', 'minmax', 'robust', 'none']

    # Loop through each normalization strategy independently
    # This ensures fair comparison across preprocessing choices
    for norm_method in normalization_methods:
        print(f"\n=== NORMALIZATION: {norm_method.upper()} ===")

        # Evaluate all selected models under the current normalization
        results = _evaluate_models(
            X_train,
            y_train,
            splits_outer,
            splits_inner,
            norm_method,
            models_to_run,
            results
        )

    return results


def _evaluate_models(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        splits_outer: int,
        splits_inner: int,
        normalization_method: str,
        models_to_run: List[str],
        results: Dict
) -> Dict:
    """
    Evaluates models using nested cross-validation.

    :param X_train: Training features
    :param y_train: Training target
    :param splits_outer: Outer CV splits
    :param splits_inner: Inner CV splits
    :param normalization_method: Normalization method
    :param models_to_run: Models to evaluate
    :param results: Results dictionary
    :return: Updated results dictionary
    """

    # Default to evaluating all models if none are specified
    if models_to_run is None:
        models_to_run = [KNN, SVM, RF, LR]

    # Map normalization names to actual sklearn scaler objects
    # 'none' means no preprocessing step will be applied
    scaler_dict = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler(),
        'none': None
    }

    # Retrieve the scaler corresponding to the selected method
    scaler = scaler_dict[normalization_method]

    # Build pipelines explicitly
    # Each pipeline ensures that scaling (if any) is applied inside CV folds (no data leakage)
    knn_pipeline = _build_pipeline(KNeighborsRegressor(), KNN, scaler)
    svm_pipeline = _build_pipeline(SVR(), SVM, scaler)
    lr_pipeline = _build_pipeline(linear_model.LinearRegression(), LR, scaler)

    # Define models and their respective hyperparameter grids
    # Each model is evaluated independently using nested CV
    model_dict = {
        KNN: {
            ESTIMATOR: knn_pipeline,
            PARAM_GRID: [{f'{KNN}__n_neighbors': list(range(1, 10))}]
        },
        SVM: {
            ESTIMATOR: svm_pipeline,
            PARAM_GRID: [{f'{SVM}__C': [0.1, 1, 10]}]
        },
        RF: {
            ESTIMATOR: RandomForestRegressor(random_state=RANDOM_SEED),
            PARAM_GRID: [{"n_estimators": [50, 100]}]
        },
        LR: {
            ESTIMATOR: lr_pipeline,
            PARAM_GRID: [{}]  # No hyperparameters for linear regression
        }
    }

    # Filter only the models requested by the user
    selected_models = {k: v for k, v in model_dict.items() if k in models_to_run}

    # Iterate through each model and evaluate it
    for model_name, param_dict in selected_models.items():
        print(f"\n--- Model: {model_name} ---")

        # Run nested cross-validation:
        # - inner loop: hyperparameter tuning
        # - outer loop: unbiased performance estimation
        info_df = nested_cross_val(
            X_train,
            y_train,
            estimator=param_dict[ESTIMATOR],
            param_grid=param_dict[PARAM_GRID],
            splits_outer=splits_outer,
            splits_inner=splits_inner
        )

        # Store results for later comparison
        results = _track_results(info_df, model_name, normalization_method, results)

    return results


def _build_pipeline(model, name: str, scaler) -> Pipeline:
    """
    Builds a pipeline with optional scaler.

    :param model: ML model
    :param name: Step name
    :param scaler: Scaler object or None
    :return: sklearn Pipeline
    """

    # If no scaler is provided, return a pipeline with only the model
    if scaler is None:
        return Pipeline([(name, model)])

    # Otherwise, include scaling step before the model
    # This ensures preprocessing happens inside CV folds
    return Pipeline([
        (SCALER, scaler),
        (name, model)
    ])


def _track_results(info_df: pd.DataFrame, model_name: str, normalization_method: str, results: Dict) -> Dict:
    """
    Stores model results.

    :param info_df: DataFrame from nested CV
    :param model_name: Model name
    :param normalization_method: Normalization used
    :param results: Results dictionary
    :return: Updated results dictionary
    """

    # Extract aggregated metrics from nested CV output
    # Values are identical across rows, so we take the first one
    result_entry = {
        'model_name': model_name,
        'normalization_method': normalization_method,
        'estimator_avg_r2': info_df['estimator_avg_r2'].iloc[0],
        'estimator_std_r2': info_df['estimator_std_r2'].iloc[0],
        'estimator_avg_mae': info_df['estimator_avg_mae'].iloc[0],
        'estimator_std_mae': info_df['estimator_std_mae'].iloc[0],
        'best_params_summary': _extract_best_params(info_df)
    }

    # Store full history of all runs (for later analysis)
    results['all_results'].append(result_entry)

    return results


def _extract_best_params(info_df: pd.DataFrame) -> str:
    """
    Extracts the hyperparameters corresponding to the highest outer-fold R².

    :param info_df: DataFrame from nested CV.
    :return: String with best parameters.
    """

    # Metric columns to ignore
    metric_cols = [
        'inner_r2', 'inner_mae',
        'fold_r2', 'fold_mae',
        'estimator_avg_r2', 'estimator_std_r2',
        'estimator_avg_mae', 'estimator_std_mae'
    ]

    # Hyperparameter columns
    params_cols = [col for col in info_df.columns if col not in metric_cols]

    # Row with highest outer R²
    best_row = info_df.loc[info_df['fold_r2'].idxmax()]

    # Extract hyperparameters
    best_params = {col: best_row[col] for col in params_cols}

    return str(best_params)

import matplotlib.pyplot as plt

def _plot_target_distribution_by_fold(fold_target_data: List[Dict], target_column: str) -> None:
    """
    Plots the distribution of the target variable across folds, comparing
    training and validation sets side by side. Useful to visually confirm
    that stratified k-fold produced folds with similar target distributions.

    :param fold_target_data: list of dicts, each containing 'fold', 'train_values'
        and 'val_values' (arrays of target values for that fold)
    :param target_column: name of the target column (used for the y-axis label)
    :return: None
    """

    n_folds = len(fold_target_data)

    # Build data and labels for a grouped boxplot:
    # for each fold, one box for train and one box for validation
    box_data = []
    box_labels = []
    box_colors = []

    for fold_info in fold_target_data:
        box_data.append(fold_info["train_values"])
        box_labels.append(f"Fold {fold_info['fold']}\nTrain")
        box_colors.append("#4C72B0")

        box_data.append(fold_info["val_values"])
        box_labels.append(f"Fold {fold_info['fold']}\nVal")
        box_colors.append("#DD8452")

    fig, ax = plt.subplots(figsize=(2.2 * n_folds, 5))

    bp = ax.boxplot(box_data, patch_artist=True, labels=box_labels)

    # Color train boxes and val boxes differently
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel(target_column)
    ax.set_title(f"{target_column} distribution across folds (Train vs Validation)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()