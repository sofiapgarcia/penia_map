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
from constants import groups
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from constants import CATEGORICAL_FEATURES , PENIA_MAP_GROUPS
from scipy import stats

# ------------------------------------------------------------------------------------------------------------------- #
# file specific constants
# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def test_normality(df, column, alpha=0.05, return_result=False, verbose=True):
    """
    Test whether a given column in a DataFrame follows a normal distribution using the Shapiro-Wilk test.

    :param df: Input DataFrame
    :param column: Name of the column to test
    :param alpha: Significance level (default is 0.05)
    :param return_result: If True, returns the result dictionary (default is False)
    :param verbose: If True, prints detailed results (default is True)
    :return: Optional dictionary with test statistic, p-value, and normality decision
    """

    # Remove missing values
    data = df[column].dropna()

    # Shapiro-Wilk test
    stat, p_value = stats.shapiro(data)
    is_normal = p_value > alpha

    if verbose:
        print(f"\nNormality Test (Shapiro-Wilk) for column: {column}")
        print(f"Statistic (W): {stat:.4f}")
        print(f"P-value: {p_value:.4f}")
        if is_normal:
            print(f"Conclusion: Data appears NORMAL (p-value > {alpha})")
        else:
            print(f"Conclusion: Data is NOT normal (p-value <= {alpha})")

    if return_result:
        return {
            "statistic": stat,
            "p_value": p_value,
            "is_normal": is_normal
        }


def correlation_with_target(df, target_col="Estimated metabolic age", alpha=0.05, top_n=None, ascending=False, verbose=True):
    """
    Compute correlation between numeric features and a target column.
    Automatically chooses Pearson (if both variables are normal) or Spearman (otherwise).

    :param df: Input DataFrame
    :param target_col: Name of the target column
    :param alpha: Significance level for normality test (default is 0.05)
    :param top_n: Number of top features to return (optional)
    :param ascending: Sort correlations in ascending order if True
    :param verbose: If True, prints detailed results
    :return: DataFrame with features, correlation values, and method used
    """

    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in DataFrame.")

    # Only numeric columns
    numeric_df = df.select_dtypes(include='number')

    if verbose:
        print(f"\nAnalyzing correlations with target: {target_col}")

    # Test normality for target
    target_result = test_normality(numeric_df, target_col, alpha, return_result=True, verbose=verbose)
    target_is_normal = target_result["is_normal"]

    results = []

    for col in numeric_df.columns:
        if col == target_col:
            continue

        # Test normality for feature
        feature_result = test_normality(numeric_df, col, alpha, return_result=True, verbose=verbose)
        feature_is_normal = feature_result["is_normal"]

        # Choose method
        if target_is_normal and feature_is_normal:
            method = "pearson"
        else:
            method = "spearman"

        # Compute correlation
        corr_value = numeric_df[[col, target_col]].corr(method=method).iloc[0, 1]

        if verbose:
            print(f"\nColumn: {col}")
            print(f"Method used: {method.upper()}")
            print(f"Correlation: {corr_value:.4f}")

        results.append({
            "feature": col,
            "correlation": corr_value
        })

    results_df = pd.DataFrame(results)

    # Sort results
    results_df = results_df.sort_values(by="correlation", ascending=ascending)

    # Return only top N features if specified
    if top_n is not None:
        results_df = results_df.head(top_n)

    return results_df


def correlation_by_groups(
    df,
    groups_dict=PENIA_MAP_GROUPS,
    target_col="Estimated metabolic age",
    alpha=0.05,
    ascending=False,
    verbose=True
):
    """
    Compute correlation between grouped features and a target column.

    :param df: Input DataFrame
    :param groups_dict: Dictionary with groups of features
    :param target_col: Target variable (default: Estimated metabolic age)
    :param alpha: Significance level for normality test
    :param ascending: Sort correlations
    :param verbose: Print results
    :return: dict -> {group_name: DataFrame with correlations}
    """

    # Check if target exists
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in DataFrame.")

    # Keep only numeric columns
    numeric_df = df.select_dtypes(include='number')

    # Test normality of target once
    target_result = test_normality(numeric_df, target_col, alpha, return_result=True, verbose=verbose)
    target_is_normal = target_result["is_normal"]

    # Dictionary to store results by group
    group_results = {}

    # Iterate over groups
    for group_name, features in groups_dict.items():

        if verbose:
            print(f"GROUP: {group_name}")

        results = []

        # Iterate over features in group
        for col in features:

            # Skip if column does not exist or is not numeric
            if col not in numeric_df.columns:
                continue

            # Test normality of feature
            feature_result = test_normality(numeric_df, col, alpha, return_result=True, verbose=verbose)
            feature_is_normal = feature_result["is_normal"]

            # Choose correlation method
            if target_is_normal and feature_is_normal:
                method = "pearson"
            else:
                method = "spearman"

            # Compute correlation
            corr_value = numeric_df[[col, target_col]].corr(method=method).iloc[0, 1]

            if verbose:
                print(f"\nFeature: {col}")
                print(f"Method: {method.upper()}")
                print(f"Correlation: {corr_value:.4f}")

            results.append({
                "feature": col,
                "correlation": corr_value
            })

        # Convert results to DataFrame
        group_df = pd.DataFrame(results)

        # Sort if not empty
        if not group_df.empty:
            group_df = group_df.sort_values(by="correlation", ascending=ascending)

        # Store result
        group_results[group_name] = group_df

    return group_results

def select_top_features_by_group(
    df,
    groups_dict=PENIA_MAP_GROUPS,
    n_features_per_group=2,
    selected_groups=["Osteopenia", "Powerpenia", "Dynapenia", "Sarcopenia", "Body Fat"],
    target_col="Estimated metabolic age",
    alpha=0.05,
    verbose=True
):
    """
    Select top N features per group based on absolute correlation with target.

    :param df: Input DataFrame
    :param groups_dict: Dictionary with feature groups
    :param n_features_per_group: Number of features to keep per group
    :param selected_groups: List of groups to consider
    :param target_col: Target variable
    :param alpha: Significance level for normality test
    :param verbose: Print progress
    :return: Filtered DataFrame with selected features + target
    """

    # Compute correlations by group
    group_corrs = correlation_by_groups(
        df,
        groups_dict,
        target_col=target_col,
        alpha=alpha,
        ascending=False,
        verbose=False  # avoid too much logging here
    )

    selected_features = []

    # Iterate over selected groups
    for group in selected_groups:

        if group not in group_corrs:
            continue

        group_df = group_corrs[group]

        if group_df.empty:
            continue

        # Create absolute correlation column
        group_df["abs_corr"] = group_df["correlation"].abs()

        # Sort by absolute correlation (descending)
        group_df = group_df.sort_values(by="abs_corr", ascending=False)

        # Select top N features
        top_features = group_df.head(n_features_per_group)["feature"].tolist()

        if verbose:
            print(f"\nGroup: {group}")
            print(f"Selected features: {top_features}")

        selected_features.extend(top_features)

    # Remove duplicates (just in case)
    selected_features = list(set(selected_features))

    # Always include target column
    final_columns = selected_features + [target_col]

    # Filter DataFrame
    filtered_df = df[final_columns]

    return filtered_df


def group_correlation_summary(correlations_df):
    """
    Compute summary statistics of correlations per variable group.

    :param correlations_df: DataFrame with index as feature names and a column
                            'Spearman Correlation'
    :return: DataFrame with correlation statistics per group
    """

    # Convert correlations to dictionary for faster lookup
    corr_dict = correlations_df["Spearman Correlation"].to_dict()

    results = []

    for group, variables in groups.items():
        values = []

        # Collect correlations for variables in this group
        for var in variables:
            if var in corr_dict:
                values.append(corr_dict[var])

        # Skip groups with no matching variables
        if len(values) == 0:
            continue

        values = np.array(values)

        # Compute statistics using absolute correlation values
        mean_abs_corr = np.mean(np.abs(values))

        results.append({
            "group": group,
            "mean_abs_correlation": mean_abs_corr,
            "mean_correlation": np.mean(values),
            "max_correlation": np.max(values),
            "min_correlation": np.min(values),
            "std_correlation": np.std(values),
            "num_variables": len(values)
        })

    # Create final DataFrame
    results_df = pd.DataFrame(results)

    # Sort by absolute mean correlation (descending)
    results_df = results_df.sort_values(by="mean_abs_correlation", ascending=False)

    return results_df



def normalize_dataset(df, categorical_prefixes=CATEGORICAL_FEATURES, scaler_type="standard"):
    """
    Normalize numeric features in a dataset while excluding one-hot encoded categorical variables
    based on their column name prefixes.

    :param df: Input DataFrame
    :param categorical_prefixes: List of prefixes identifying one-hot encoded categorical columns to exclude
    :param scaler_type: Type of scaler to use ('standard', 'minmax', 'robust', 'maxabs')
    :return: DataFrame with normalized numeric features and unchanged categorical features
    """

    # Copy dataframe to avoid modifying original data
    df_normalized = df.copy()

    if categorical_prefixes is None:
        categorical_prefixes = []

    # Dictionary of available scalers
    scaler_dict = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
        "maxabs": MaxAbsScaler()
    }

    if scaler_type not in scaler_dict:
        raise ValueError(f"Unsupported scaler_type '{scaler_type}'. Choose from {list(scaler_dict.keys())}")

    scaler = scaler_dict[scaler_type]

    # Identify columns to exclude (categorical one-hot encoded)
    cols_to_exclude = [
        col for col in df.columns
        if any(col.startswith(prefix) for prefix in categorical_prefixes)
    ]

    # Select columns to normalize
    cols_to_normalize = [
        col for col in df.columns
        if col not in cols_to_exclude
    ]

    # Apply scaling
    df_normalized[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])

    return df_normalized

def inspect_pca_variance(df, target_col="Estimated metabolic age"):
    """
    Fit PCA on numeric features (excluding target) and plot explained variance
    to help manually choose the number of components.

    :param df: Input DataFrame (numeric features only or mixed)
    :param target_col: Name of the target column to exclude
    :return: fitted PCA object, cumulative explained variance array
    """

    # Drop target column if present
    X = df.drop(columns=[target_col], errors='ignore')

    # Select numeric features only
    X = X.select_dtypes(include='number')

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit PCA with all components
    pca = PCA()
    pca.fit(X_scaled)

    # Compute explained variance
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    # Plot cumulative variance
    plt.figure()
    plt.plot(cumulative, marker='o')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA - Cumulative Explained Variance")
    plt.grid()
    plt.show()

    return pca, cumulative


def apply_pca(df, n_components, target_col="Estimated metabolic age"):
    """
    Apply PCA with a predefined number of components, excluding the target variable.

    :param df: Input DataFrame (numeric features only or mixed)
    :param n_components: Number of principal components to keep
    :param target_col: Name of the target column to exclude
    :return: fitted PCA object, transformed DataFrame, loadings DataFrame
    """

    # Drop target column if present
    X = df.drop(columns=[target_col], errors='ignore')

    # Select numeric features
    X = X.select_dtypes(include='number')

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Create loadings DataFrame
    loadings = pd.DataFrame(
        pca.components_.T,
        index=X.columns,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )

    # Create transformed DataFrame
    X_pca_df = pd.DataFrame(
        X_pca,
        columns=loadings.columns,
        index=df.index
    )

    return pca, X_pca_df, loadings



