"""
Functions to...

Available Functions
-------------------
[Public]
function1(...):

[Private]
_function2(...):
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
from scipy.stats import stats
import numpy as np

from constants import *


# internal imports

# ------------------------------------------------------------------------------------------------------------------- #
# file specific constants
# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def compute_normality_map(df, alpha=0.05):
    """
    Compute a mapping of numeric columns to normality status using the Shapiro-Wilk test.

    This function is used to determine whether each feature follows a normal distribution,
    which later controls whether Pearson or Spearman correlation is applied.

    :param df: Input DataFrame
    :param alpha: Significance level for normality test
    :return: Dictionary mapping column names to boolean values (True if normally distributed)
    """

    numeric_df = df.select_dtypes(include="number")

    normality_map = {}

    for col in numeric_df.columns:
        data = numeric_df[col].dropna()

        # Shapiro-Wilk test for normality
        stat, p_value = stats.shapiro(data)

        # Determine normality based on p-value threshold
        normality_map[col] = p_value > alpha

    return normality_map

def compute_correlation(df, feature_col, target_col, normality_map):
    """
    Compute correlation between a feature and the target variable using an adaptive method.

    Pearson correlation is used if both variables are normally distributed;
    otherwise, Spearman correlation is applied.

    :param df: Input DataFrame
    :param feature_col: Name of the feature column
    :param target_col: Name of the target column
    :param normality_map: Dictionary indicating normality of each variable
    :return: Tuple (correlation value, correlation method used)
    """

    # Select correlation method based on normality assumptions
    if normality_map.get(feature_col, False) and normality_map.get(target_col, False):
        method = "pearson"
    else:
        method = "spearman"

    # Compute correlation
    corr_value = df[[feature_col, target_col]].corr(method=method).iloc[0, 1]

    return corr_value, method

def remove_collinear_features(features, df, threshold=0.85, verbose=False):
    """
    Remove highly correlated features within a group to reduce multicollinearity.

    This step ensures that redundant features (highly correlated with each other)
    are removed before computing correlation with the target variable.

    :param features: List of feature names belonging to a group
    :param df: Input DataFrame
    :param threshold: Correlation threshold above which features are considered redundant
    :param verbose: If True, prints removed features
    :return: Tuple (filtered_features, removed_features)
    """

    if len(features) <= 1:
        return features, []

    # Compute absolute correlation matrix
    corr_matrix = df[features].corr().abs()

    # Upper triangle of correlation matrix (avoid duplicate comparisons)
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Identify features to drop based on correlation threshold
    to_drop = [
        col for col in upper.columns
        if any(upper[col] > threshold)
    ]

    # Keep only non-redundant features
    kept_features = [f for f in features if f not in to_drop]

    if verbose:
        print(f"Removed due to collinearity: {to_drop}")

    return kept_features, to_drop

def select_top_features_by_group(
    df,
    groups_dict=PENIA_MAP_GROUPS,
    selected_groups=["Osteopenia", "Powerpenia", "Dynapenia", "Sarcopenia", "Body Fat"],
    target_col="Estimated metabolic age",
    n_features_per_group=2,
    corr_threshold=0.85,
    alpha=0.05,
    verbose=True
):
    """
    Select top N features per group based on adaptive correlation with the target variable.

    This function:
    - Removes highly correlated features within each group
    - Uses Shapiro-Wilk test to choose Pearson or Spearman correlation
    - Selects the most relevant features per group
    - Returns a filtered DataFrame with selected features and target column

    :param df: Input DataFrame
    :param groups_dict: Dictionary mapping group names to feature lists
    :param selected_groups: List of groups to include in selection
    :param target_col: Target variable name
    :param n_features_per_group: Number of features to select per group
    :param corr_threshold: Threshold for removing collinear features
    :param alpha: Significance level for normality test
    :param verbose: If True, prints selection process details
    :return: Filtered DataFrame with selected features and target column
    """

    if verbose:
        print("\nFEATURE SELECTION PIPELINE\n")

    # Step 1: Compute normality map once
    normality_map = compute_normality_map(df, alpha=alpha)

    selected_features = []

    # Step 2: Iterate through selected groups
    for group in selected_groups:

        if group not in groups_dict:
            continue

        raw_features = groups_dict[group]
        raw_features = [f for f in raw_features if f in df.columns]

        if len(raw_features) == 0:
            continue

        if verbose:
            print(f"\nGROUP: {group}")
            print(f"Initial features ({len(raw_features)}): {raw_features}")

        # Step 3: Remove multicollinearity within group
        filtered_features, dropped = remove_collinear_features(
            raw_features,
            df,
            threshold=corr_threshold,
            verbose=False
        )

        if verbose:
            print(f"Dropped (collinearity): {dropped}")
            print(f"Remaining features: {filtered_features}")

        # Step 4: Compute correlation with target
        scored_features = []

        for feature in filtered_features:

            corr_value, method = compute_correlation(
                df,
                feature,
                target_col,
                normality_map
            )

            scored_features.append((feature, corr_value))

            if verbose:
                print(f"{feature} -> {corr_value:.4f} ({method})")

        # Step 5: Rank features
        scored_features.sort(key=lambda x: abs(x[1]), reverse=True)

        top_features = [f for f, _ in scored_features[:n_features_per_group]]

        selected_features.extend(top_features)

        if verbose:
            print(f"\nSelected for {group}: {top_features}")

    # Step 6: Final dataset
    final_columns = selected_features + [target_col]

    if verbose:
        print("\n================ FINAL SELECTION ================")
        print(f"Total selected features: {len(selected_features)}")
        print(f"Features: {selected_features}")
        print("================================================\n")

    return df[final_columns]

# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #