import numpy as np
import pandas as pd
import smogn

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import r2_score



def get_scaler(scaler_type):
    """
    Return a scaler object based on the specified type.

    :param scaler_type: Type of scaler to create ('standard', 'minmax', 'quantile')
    :return: Initialized scaler object
    """

    if scaler_type == "standard":
        scaler = StandardScaler()

    elif scaler_type == "minmax":
        scaler = MinMaxScaler()

    elif scaler_type == "quantile":
        # Produces approximately Gaussian-distributed features
        scaler = QuantileTransformer(output_distribution="normal")

    else:
        raise ValueError("Unknown scaler_type")

    return scaler

def get_feature_selector(method, n_features, score_func=f_regression):
    """
    Return a feature selection or dimensionality reduction object.

    :param method: Feature selection method ('kbest' or 'pca')
    :param n_features: Number of features/components to retain
    :param score_func: Scoring function for SelectKBest (only used if method='kbest'), example: f_regression, mutual_info_regression and r_regression
    :return: Feature selector instance
    """
    if method == "kbest":
        selector = SelectKBest(score_func=score_func, k=n_features)

    elif method == "pca":
        selector = PCA(n_components=n_features)

    else:
        raise ValueError("Unknown feature_selection method")

    return selector

def get_model_and_param_grid(model_type, random_state):
    """
    Return model instance and corresponding hyperparameter grid.

    :param model_type: Model type ('rf' or 'svr')
    :type model_type: str
    :param random_state: Random seed
    :type random_state: int
    :return: Tuple containing model instance and parameter grid
    :rtype: tuple
    """


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

    result = (model, param_grid)

    return result

def build_pipeline(scaler, selector, model):
    """
    Construct a sklearn Pipeline.

    :param scaler: Preprocessing scaler
    :type scaler: sklearn Transformer
    :param selector: Feature selector
    :type selector: sklearn Transformer
    :param model: Regression model
    :type model: sklearn estimator
    :return: Configured Pipeline
    :rtype: sklearn.pipeline.Pipeline
    """

    pipeline = Pipeline(
        steps=[
            ("scaler", scaler),
            ("feature_selection", selector),
            ("model", model),
        ]
    )

    return pipeline

def apply_smogn(train_df, target_column, target_size, random_state):
    """
    Apply SMOGN augmentation to the training data.

    :param train_df: Training dataset
    :param target_column: Target column name
    :param target_size: Maximum size after augmentation
    :param random_state: Random seed
    :return: Augmented dataset
    """

    augmented = smogn.smoter(
        data=train_df,
        y=target_column,
        samp_method="balance",
    )

    if len(augmented) > target_size: #TODO: ver se faz sentido e fazer print do numero final depois de aplicar esta técnica
        augmented = augmented.sample(target_size, random_state=random_state)

    return augmented

def evaluate_fold(
    train_df,
    test_df,
    target_column,
    pipeline,
    param_grid,
    target_size,
    random_state,
):
    """
    Train and evaluate one outer CV fold.

    :return: R² score for the fold
    :rtype: float
    """

    train_aug = apply_smogn(
        train_df,
        target_column,
        target_size,
        random_state,
    )

    X_train = train_aug.drop(columns=target_column).values
    y_train = train_aug[target_column].values

    X_test = test_df.drop(columns=target_column).values
    y_test = test_df[target_column].values

    inner_cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=inner_cv,
        scoring="r2",
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    y_pred = grid_search.predict(X_test)

    score = r2_score(y_test, y_pred)

    return score

def nested_cv_regression_pipeline_smogn(
    df,
    target_column="Estimated metabolic age",
    model_type="rf",
    scaler_type="standard",
    feature_selection="kbest",
    n_features=20,
    target_size=100,
    random_state=42,
):
    """
    Perform nested cross-validation with SMOGN augmentation.

    :return: Mean outer CV R² score
    :rtype: float
    """

    scaler = get_scaler(scaler_type)
    selector = get_feature_selector(feature_selection, n_features)
    model, param_grid = get_model_and_param_grid(model_type, random_state)

    pipeline = build_pipeline(scaler, selector, model)

    outer_cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

    scores = []

    for train_idx, test_idx in outer_cv.split(df):

        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        score = evaluate_fold(
            train_df,
            test_df,
            target_column,
            pipeline,
            param_grid,
            target_size,
            random_state,
        )

        scores.append(score)

    mean_score = np.mean(scores)

    return mean_score