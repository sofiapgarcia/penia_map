from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, VarianceThreshold
from collections import Counter
import pandas as pd


def feature_selection_filters(
        df: pd.DataFrame,
        target_col: str = "Estimated metabolic age",
        max_features: int = 20
) -> pd.DataFrame:
    """
    Select features using multiple filter-based methods, print selected features for each,
    and return a DataFrame containing only the selected features plus the target.

    :param df: Input DataFrame
    :param target_col: Name of the target column
    :param max_features: Maximum number of features to select based on consensus
    :return: DataFrame with selected features and target column
    """

    # ---- Separate features and target ----
    X = df.drop(columns=[target_col])
    y = df[target_col]

    results = {}

    # Variance Threshold
    selector_var = VarianceThreshold(threshold=0.01)
    selector_var.fit(X)
    selected_var = X.columns[selector_var.get_support()]
    results['VarianceThreshold'] = selected_var.tolist()[:max_features]

    # F-test (regression)
    selector_f = SelectKBest(score_func=f_regression, k=min(max_features, X.shape[1]))
    selector_f.fit(X, y)
    selected_f = X.columns[selector_f.get_support()]
    results['F_regression'] = selected_f.tolist()

    # Mutual Information
    selector_mi = SelectKBest(score_func=mutual_info_regression, k=min(max_features, X.shape[1]))
    selector_mi.fit(X, y)
    selected_mi = X.columns[selector_mi.get_support()]
    results['Mutual_info'] = selected_mi.tolist()

    # Correlation
    corr = X.corrwith(y).abs().sort_values(ascending=False)
    selected_corr = corr.head(max_features).index.tolist()
    results['Correlation'] = selected_corr

    # ---- Print selected features per method ----
    print("Selected features by method:\n" + "-" * 40)
    for method, features in results.items():
        print(f"{method}: {features}")
    print('-' * 40)

    # ---- Count frequency across methods ----
    all_selected = [feat for feats in results.values() for feat in feats]
    feature_counts = Counter(all_selected)

    # ---- Select top features by consensus ----
    ranked_features = [
        feat for feat, _ in feature_counts.most_common(max_features)
    ]

    # ---- Print ranking ----
    print(f"Top {max_features} features based on selection frequency:\n" + "-" * 40)
    for feat in ranked_features:
        print(feat)

    # ---- Build final DataFrame ----
    df_selected = df[ranked_features + [target_col]]

    return df_selected

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    QuantileTransformer,
    OneHotEncoder,
)
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import r2_score


# -----------------------------------------------------------
# Data augmentation for small tabular regression datasets
# -----------------------------------------------------------
def augment_tabular_regression(X, y, target_size=100, noise_std=0.01, random_state=42):
    """
    Generate synthetic samples for tabular regression by interpolating
    between random pairs of samples and adding Gaussian noise.

    Parameters
    ----------
    X : array-like (n_samples, n_features)
    y : array-like (n_samples,)
    target_size : int
        Desired number of samples after augmentation
    noise_std : float
        Standard deviation of Gaussian noise
    random_state : int

    Returns
    -------
    X_aug, y_aug : augmented dataset
    """

    rng = np.random.default_rng(random_state)

    n_samples = X.shape[0]

    if n_samples >= target_size:
        return X, y

    n_to_generate = target_size - n_samples

    X_syn = []
    y_syn = []

    for _ in range(n_to_generate):
        i, j = rng.integers(0, n_samples, 2)

        alpha = rng.random()

        x_new = alpha * X[i] + (1 - alpha) * X[j]
        y_new = alpha * y[i] + (1 - alpha) * y[j]

        x_new += rng.normal(0, noise_std, size=X.shape[1])

        X_syn.append(x_new)
        y_syn.append(y_new)

    X_aug = np.vstack([X, np.array(X_syn)])
    y_aug = np.concatenate([y, np.array(y_syn)])

    return X_aug, y_aug


# -----------------------------------------------------------
# Main nested CV function
# -----------------------------------------------------------
def nested_cv_regression_pipeline(
    X,
    y,
    categorical_features,
    model_type="rf",
    scaler_type="standard",
    feature_selection="kbest",
    n_features=20,
    augmentation_size=100,
    random_state=42,
):
    """
    Perform nested cross-validation for regression using a configurable pipeline.

    Supports Random Forest and SVR models, multiple scaling methods,
    feature selection, and tabular data augmentation.

    Parameters
    ----------
    X : array-like
        Feature matrix.
    y : array-like
        Target vector.
    categorical_features : list
        Indices of categorical variables.
    model_type : str
        'rf' or 'svr'.
    scaler_type : str
        'standard', 'minmax', or 'quantile'.
    feature_selection : str
        'kbest' or 'pca'.
    n_features : int
        Number of features after selection.
    augmentation_size : int
        Number of samples after augmentation.
    random_state : int

    Returns
    -------
    float
        Mean outer CV R² score.
    """

    # -------------------------
    # Choose scaler
    # -------------------------
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type == "quantile":
        scaler = QuantileTransformer(output_distribution="normal")
    else:
        raise ValueError("Unknown scaler_type")

    # -------------------------
    # Feature selection
    # -------------------------
    if feature_selection == "kbest":
        selector = SelectKBest(score_func=f_regression, k=n_features)
    elif feature_selection == "pca":
        selector = PCA(n_components=n_features)
    else:
        raise ValueError("Unknown feature_selection")

    # -------------------------
    # Model + hyperparameters
    # -------------------------
    if model_type == "rf":
        model = RandomForestRegressor(random_state=random_state)

        param_grid = {
            "model__n_estimators": [50, 100],
            "model__max_depth": [None, 10, 20],
        }

    elif model_type == "svr":
        model = SVR()

        param_grid = {
            "model__C": [0.1, 1, 10],
            "model__epsilon": [0.01, 0.1, 1],
            "model__kernel": ["rbf", "linear"],
        }

    else:
        raise ValueError("Unknown model_type")

    # -------------------------
    # Preprocess categorical features
    # -------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            (
                "num",
                "passthrough",
                [i for i in range(X.shape[1]) if i not in categorical_features],
            ),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("scaler", scaler),
            ("feature_selection", selector),
            ("model", model),
        ]
    )

    # -------------------------------------------------------
    # Nested Cross Validation
    # -------------------------------------------------------
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

    outer_scores = []

    for train_idx, test_idx in outer_cv.split(X):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # ✅ Data augmentation ONLY on training fold
        X_train_aug, y_train_aug = augment_tabular_regression(
            X_train,
            y_train,
            target_size=augmentation_size,
            random_state=random_state,
        )

        inner_cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=inner_cv,
            scoring="r2",
            n_jobs=-1,
        )

        grid_search.fit(X_train_aug, y_train_aug)

        y_pred = grid_search.predict(X_test)

        score = r2_score(y_test, y_pred)
        outer_scores.append(score)

    return np.mean(outer_scores)