"""
Functions to build, evaluate, and apply DXA prediction models
using TANITA body composition measurements.

The module implements:
- Adaptive correlation analysis
- Feature selection
- Ordinary Least Squares (OLS) regression
- DXA variable prediction
- R² performance visualization

Available Functions
-------------------
[Public]
build_dxa_predictor(...): Build statistical regression models to estimate DXA variables from TANITA measurements.
predict_dxa(...): Predict DXA variables for new TANITA samples using previously trained regression models.
plot_r2_distribution_histogram(...): Plot the distribution of R² values across all generated models.
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import itertools
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# internal imports
from feature_reduction import adaptive_correlation_matrix

# ------------------------------------------------------------------------------------------------------------------- #
# file specific constants
# ------------------------------------------------------------------------------------------------------------------- #

DEFAULT_TANITA_COLUMNS = [
    "Body mass(kg)",
    "Body mass index (BMI)",
    "Global fat %",
    "Arm fat (right) %",
    "Arm fat (left) %",
    "Leg fat (right) %",
    "Leg fat (left) %",
    "Torso fat %",
    "Global muscle kg",
    "Arm muscle (right) kg",
    "Arm muscle (left) kg",
    "Leg muscle (right) kg",
    "Leg muscle (left) kg",
    "Torso muscle kg",
    "Estimated bone mass kg",
    "Visceral fat rating",
    "Daily calorie intake (DCI)",
    "Estimated metabolic age",
    "Global body water %",
    "Altura_testes"
]

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def build_dxa_predictor(
    df,
    tanita_cols=DEFAULT_TANITA_COLUMNS,
    corr_threshold=0.30,
    strong_corr_threshold=0.85,
    max_features=5,
    alpha=0.05
):
    """
    Build statistical regression models to predict DXA variables
    using TANITA measurements.

    The function:
    - Automatically identifies DXA variables
    - Computes adaptive correlations
    - Uses a single-feature regression when correlation is very strong
    - Otherwise tests all feature combinations
    - Fits OLS regression models
    - Selects the best model using adjusted R²

    :param df: Input DataFrame
    :param tanita_cols: TANITA predictor columns
    :param corr_threshold: Minimum absolute correlation required
    :param strong_corr_threshold: Correlation threshold for single-feature models
    :param max_features: Maximum number of features per model
    :param alpha: Significance level for adaptive correlation
    :return: Dictionary containing best model per DXA variable
    """

    # Validate TANITA columns
    missing_cols = [
        col for col in tanita_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing TANITA columns: {missing_cols}"
        )
    ignore_cols = ["ID", "Idade", "Altura"]

    # Automatically identify DXA variables
    dxa_cols = [
        col for col in df.select_dtypes(include="number").columns
        if col not in tanita_cols
           and col not in ignore_cols
    ]

    results = {}

    for target in dxa_cols:

        print(f"Processing target: {target}")

        # -----------------------------------
        # Compute adaptive correlations
        # -----------------------------------
        subset_df = df[tanita_cols + [target]].copy()

        corr_matrix, method_matrix = adaptive_correlation_matrix(
            subset_df,
            alpha=alpha
        )

        # Extract correlations with target
        correlations = (
            corr_matrix[target]
            .drop(target)
            .abs()
            .sort_values(ascending=False)
        )

        best_feature = correlations.index[0]
        best_corr = correlations.iloc[0]

        # -----------------------------------
        # Use single-feature model if
        # correlation is very strong
        # -----------------------------------
        if best_corr >= strong_corr_threshold:

            X = df[[best_feature]]
            y = df[target]

            X_const = sm.add_constant(X, has_constant='add')

            model = sm.OLS(y, X_const).fit()

            results[target] = {
                "model": model,
                "features": [best_feature],
                "r2": float(model.rsquared),
                "adjusted_r2": float(model.rsquared_adj),
                "n_features": 1,
                "correlation_value": float(best_corr),
                "correlation_method": method_matrix.loc[
                    best_feature,
                    target
                ],
                "selection_method": "single_feature_correlation"
            }
            print("r2:",model.rsquared)
            print("features:", best_feature)

            continue

        # -----------------------------------
        # Filter relevant features
        # -----------------------------------
        candidate_features = correlations[
            correlations >= corr_threshold
        ].index.tolist()

        # Fallback if no feature passes threshold
        if len(candidate_features) == 0:
            candidate_features = [best_feature]

        # Limit maximum feature count
        max_k = min(max_features, len(candidate_features))

        best_model = None
        best_features = None
        best_adj_r2 = -999
        best_r2 = -999

        # -----------------------------------
        # Test all feature combinations
        # -----------------------------------
        for k in range(1, max_k + 1):

            combinations = itertools.combinations(
                candidate_features,
                k
            )

            for feature_subset in combinations:

                feature_subset = list(feature_subset)

                X = df[feature_subset]
                y = df[target]

                # Add intercept
                X_const = sm.add_constant(X, has_constant='add')

                # Fit OLS regression
                model = sm.OLS(y, X_const).fit()

                adj_r2 = model.rsquared_adj

                # Keep best model
                if adj_r2 > best_adj_r2:

                    best_model = model
                    best_features = feature_subset
                    best_adj_r2 = adj_r2
                    best_r2 = model.rsquared

        # -----------------------------------
        # Store best model
        # -----------------------------------
        results[target] = {
            "model": best_model,
            "features": best_features,
            "r2": float(best_r2),
            "adjusted_r2": float(best_adj_r2),
            "n_features": len(best_features),
            "correlation_value": float(best_corr),
            "correlation_method": method_matrix.loc[
                best_features[0],
                target
            ],
            "selection_method": "best_subset_regression"
        }
        print("r2:", best_r2)
        print("features:", best_features)

    return results


def predict_dxa(models, tanita_df):
    """
    Predict DXA variables from TANITA samples.

    :param models: Output from build_dxa_predictor
    :param tanita_df: DataFrame containing TANITA variables
    :return: DataFrame with original TANITA columns and predicted DXA values
    """

    predictions = {}

    for target, info in models.items():

        model = info["model"]
        features = info["features"]

        # Check if all required features exist in the input data
        missing = [f for f in features if f not in tanita_df.columns]

        X = tanita_df[features]
        X_const = sm.add_constant(X, has_constant='add')
        predictions[target] = model.predict(X_const)

    # Create output DataFrame with original TANITA columns and predictions
    result_df = tanita_df.copy()

    if predictions:
        predictions_df = pd.DataFrame(predictions)
        result_df = pd.concat([result_df, predictions_df], axis=1)

    return result_df

def plot_r2_distribution_histogram(models):
    """
    Plot R² distribution as histogram with 30 bins and statistics.

    :param models: Dictionary returned by build_dxa_predictor
    """

    # Extract R² values
    r2_values = [
        info["r2"]
        for info in models.values()
        if "r2" in info and info["r2"] is not None
    ]

    r2_series = pd.Series(r2_values)

    # Calculate statistics
    mean_r2 = r2_series.mean()
    median_r2 = r2_series.median()
    std_r2 = r2_series.std()
    min_r2 = r2_series.min()
    max_r2 = r2_series.max()
    n_models = len(r2_series)

    # Print summary statistics
    print("R² Summary Statistics:")
    print("=" * 50)
    print(f"Count:        {n_models}")
    print(f"Mean:         {mean_r2:.4f}")
    print(f"Median:       {median_r2:.4f}")
    print(f"Std Dev:      {std_r2:.4f}")
    print(f"Min:          {min_r2:.4f}")
    print(f"Max:          {max_r2:.4f}")
    print("=" * 50)
    print()

    # Create figure with larger size for better readability
    fig, ax = plt.subplots(figsize=(14, 6))

    # Histogram with 30 bins
    ax.hist(
        r2_series,
        bins=30,
        color='steelblue',
        edgecolor='darkblue',
        alpha=0.7
    )

    # Add vertical lines for mean and median
    ax.axvline(mean_r2, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_r2:.3f}')
    ax.axvline(median_r2, color='green', linestyle='--', linewidth=2, label=f'Median: {median_r2:.3f}')

    # Labels and title
    ax.set_xlabel('R²', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Models', fontsize=12, fontweight='bold')
    ax.set_title('R² Distribution across DXA models', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Set x-axis limits
    ax.set_xlim(-0.05, 1.05)

    plt.tight_layout()
    plt.show()