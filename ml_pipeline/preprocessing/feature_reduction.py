"""
Feature selection based on correlation analysis.

This module provides tools for:
- Group-wise feature selection based on adaptive correlation with a target variable
- Removal of multicollinearity within and across feature groups
- Grouped correlation analysis for exploratory data analysis

The feature selection pipeline:
- Uses Shapiro-Wilk test to decide between Pearson and Spearman correlation
- Selects top features per predefined groups
- Removes redundant features within groups and globally
- Returns a final reduced feature set ready for modeling

Available Functions
-------------------

[Public]
select_features_by_group(...): Selects top N features per group based on correlation with the target and applies intra- and inter-group collinearity removal.
correlation_by_groups(...): Computes correlation between features and target grouped by predefined feature sets.
adaptive_correlation_matrix(...): Computes a full pairwise adaptive correlation matrix (and matching p-value matrix) across numeric variables.
top_correlations_per_target(...): For each target, returns the top N variables most correlated with it, restricted to statistically significant relationships (p-value < alpha).

[Private]
_compute_normality_map(...): Computes normality status of numeric variables using the Shapiro-Wilk test.
_compute_correlation(...): Computes adaptive correlation (value, p-value, method) between two variables
_remove_collinear_features(...): Removes highly correlated features while keeping the most relevant ones based on correlation with the target variable.
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats

# internal imports
from constants import PENIA_MAP_GROUPS


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def select_features_by_group(
    df,
    groups_dict=PENIA_MAP_GROUPS,
    selected_groups=["Osteopenia", "Powerpenia", "Dynapenia", "Sarcopenia"],
    target_col="Estimated metabolic age",
    n_features_per_group=2,
    corr_threshold=0.85,
    alpha=0.05,
    global_corr_threshold=0.85,
    verbose=True
):
    """
    Select top N features per group based on adaptive correlation with the target variable.

    This function:
    - Removes highly correlated features within each group
    - Uses Shapiro-Wilk test to choose Pearson or Spearman correlation
    - Selects the most relevant features per group
    - Applies a final global collinearity filtering step across all selected features
    - Returns a filtered DataFrame with selected features and target column

    :param df: Input DataFrame
    :param groups_dict: Dictionary mapping group names to feature lists
    :param selected_groups: List of groups to include in selection
    :param target_col: Target variable name
    :param n_features_per_group: Number of features to select per group
    :param corr_threshold: Threshold for removing collinear features within groups
    :param global_corr_threshold: Threshold for removing collinear features globally
    :param alpha: Significance level for normality test
    :param verbose: If True, prints selection process details
    :return: Filtered DataFrame with selected features and target column
    """

    if verbose:
        print("\nFEATURE SELECTION PIPELINE\n")

    # Step 1: Compute normality map once
    normality_map = _compute_normality_map(df, alpha=alpha)

    selected_features = []

    # Step 2: Iterate through selected groups
    for group in selected_groups:

        if group not in groups_dict:
            continue

        raw_features = [f for f in groups_dict[group] if f in df.columns]

        if len(raw_features) == 0:
            continue

        if verbose:
            print(f"\nGROUP: {group}")
            print(f"Initial features ({len(raw_features)}): {raw_features}")

        # Step 3: Remove multicollinearity within group
        filtered_features, dropped = _remove_collinear_features(raw_features, df, normality_map=normality_map,
                                                                threshold=corr_threshold, verbose=False)

        if verbose:
            print(f"Dropped (collinearity): {dropped}")
            print(f"Remaining features: {filtered_features}")

        # Step 4: Compute correlation with target
        scored_features = []

        for feature in filtered_features:

            corr_value, p_value, method = _compute_correlation(df, feature, target_col, normality_map)

            scored_features.append((feature, corr_value))

            if verbose:
                print(f"{feature} -> {corr_value:.4f} (p={p_value:.4g}, {method})")

        # Step 5: Rank features and select top N per group
        scored_features.sort(key=lambda x: abs(x[1]), reverse=True)
        top_features = [f for f, _ in scored_features[:n_features_per_group]]

        selected_features.extend(top_features)

        if verbose:
            print(f"\nSelected for {group}: {top_features}")

    # Step 6: Global collinearity filtering (final cleanup step)
    if verbose:
        print("\n--- GLOBAL COLLINEARITY FILTER ---")

    final_features, removed_global = _remove_collinear_features(selected_features, df, normality_map=normality_map,
                                                                threshold=global_corr_threshold, verbose=verbose)

    final_columns = final_features + [target_col]

    if verbose:
        print(f"Removed globally: {removed_global}")

        print("\n================ FINAL SELECTION ================")
        print(f"Total features before global filter: {len(selected_features)}")
        print(f"Total features after global filter: {len(final_features)}")
        print(f"Features: {final_features}")
        print("================================================\n")

    return df[final_columns]



def correlation_by_groups(
    df,
    groups_dict=PENIA_MAP_GROUPS,
    target_col="Estimated metabolic age",
    alpha=0.05,
    verbose=False
):
    """
    Compute grouped correlations and return a dictionary of DataFrames per group.

    This function reuses:
    - compute_normality_map (for normality estimation)
    - compute_correlation (for adaptive Pearson/Spearman correlation)

    The output structure is identical to correlation_by_groups, but enriched
    with additional metadata for inspection and debugging.

    :param df: Input DataFrame
    :param groups_dict: Dictionary mapping group names to feature lists
    :param target_col: Target variable name
    :param alpha: Significance level for normality test
    :param verbose: If True, prints progress
    :return: Dictionary {group_name: DataFrame with correlation analysis}
    """

    # Check if target column exists in the dataset
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in DataFrame.")

    # Compute normality for all numeric variables (done once for efficiency)
    normality_map = _compute_normality_map(df, alpha=alpha)

    # Dictionary to store results per group
    group_results = {}

    # Iterate over each feature group
    for group_name, features in groups_dict.items():

        if verbose:
            print(f"\nGROUP: {group_name}")

        results = []

        # Iterate over features within the current group
        for feature in features:

            # Skip feature if it is not present in the DataFrame
            if feature not in df.columns:
                continue

            # Compute adaptive correlation using existing pipeline logic
            corr_value, p_value, method = _compute_correlation(df, feature, target_col, normality_map)

            # Store detailed results for inspection
            results.append({
                "feature": feature,
                "correlation": corr_value,
                "p_value": p_value,
                "method": method
            })

            # Optional verbose logging for debugging
            if verbose:
                print(f"{feature} -> {corr_value:.4f} (p={p_value:.4g}, {method})")

        # Convert results list into a DataFrame
        group_df = pd.DataFrame(results)

        # Sort features by absolute correlation
        if not group_df.empty:
            # Create temporary absolute correlation for sorting
            group_df["abs_corr"] = group_df["correlation"].abs()

            # Sort by absolute correlation (ranking strength)
            group_df = group_df.sort_values(
                by="abs_corr",
                ascending=False
            )

            # Remove helper column before returning output
            group_df = group_df.drop(columns=["abs_corr"])

        # Store DataFrame under group name
        group_results[group_name] = group_df

    return group_results



def adaptive_correlation_matrix(df, alpha=0.05):
    """
    Compute an adaptive correlation matrix for all numeric variables.

    This function:
    - Computes normality of each numeric variable using the Shapiro-Wilk test
    - Applies an adaptive correlation strategy (Pearson or Spearman)
      depending on the distribution of each variable pair
    - Returns the correlation matrix, the matching p-value matrix, and the
      method used per pair

    The resulting matrices allow full inspection of:
    - Correlation strength between variables
    - Statistical significance (p-value) of each pairwise correlation
    - Which statistical method was applied for each pair

    :param df: Input DataFrame
    :param alpha: Significance level for normality test
    :return:
        - corr_matrix: DataFrame with correlation values
        - pval_matrix: DataFrame with p-values for each pairwise correlation
        - method_matrix: DataFrame with correlation methods used (pearson/spearman)
    """

    # Select only numeric variables (correlation requires numeric data)
    numeric_df = df.select_dtypes(include="number")

    # Compute normality map once for efficiency
    normality_map = _compute_normality_map(numeric_df, alpha=alpha)

    # Initialize matrices to store results
    cols = numeric_df.columns
    corr_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
    pval_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
    method_matrix = pd.DataFrame(index=cols, columns=cols, dtype=object)

    # Iterate over all variable pairs
    for i, col1 in enumerate(cols):
        for j, col2 in enumerate(cols):

            # Compute only upper triangle (including diagonal)
            # and mirror results to avoid redundant computation
            if i <= j:

                corr_value, p_value, method = _compute_correlation(
                    numeric_df, col1, col2, normality_map
                )

                # Store correlation values symmetrically
                corr_matrix.loc[col1, col2] = corr_value
                corr_matrix.loc[col2, col1] = corr_value

                # Store p-values symmetrically
                pval_matrix.loc[col1, col2] = p_value
                pval_matrix.loc[col2, col1] = p_value

                # Store method used symmetrically
                method_matrix.loc[col1, col2] = method
                method_matrix.loc[col2, col1] = method

    return corr_matrix, pval_matrix, method_matrix



# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #


def _compute_normality_map(df, alpha=0.05):
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

def _compute_correlation(df, col1, col2, normality_map):
    """
    Compute correlation between two variables using an adaptive method, along
    with the p-value of that correlation.

    Pearson correlation is used if both variables are normally distributed;
    otherwise, Spearman correlation is applied.

    :param df: Input DataFrame
    :param col1: Name of the first column to compute correlation
    :param col2: Name of the second column to compute correlation
    :param normality_map: Dictionary indicating normality of each variable
    :return: Tuple (correlation value, p_value, correlation method used)
    """

    # Select correlation method based on normality assumptions
    if normality_map.get(col1, False) and normality_map.get(col2, False):
        method = "pearson"
    else:
        method = "spearman"

    # Correlation of a variable with itself (diagonal of the matrix).
    # Handled separately because df[[col1, col2]] with col1 == col2 would
    # return a DataFrame with two duplicated columns instead of a Series,
    # which breaks pearsonr/spearmanr.
    if col1 == col2:
        return 1.0, 0.0, method

    # Drop rows where either variable is missing, otherwise pearsonr/spearmanr will fail
    valid = df[[col1, col2]].dropna()

    # Compute correlation and p-value
    if method == "pearson":
        corr_value, p_value = stats.pearsonr(valid[col1], valid[col2])
    else:
        corr_value, p_value = stats.spearmanr(valid[col1], valid[col2])

    return corr_value, p_value, method

def _remove_collinear_features(features, df, normality_map, target_col="Estimated metabolic age", threshold=0.85, verbose=False):
    """
    Remove highly correlated features while keeping the one most correlated with the target.

    :param features: List of feature names belonging to a group
    :param df: Input DataFrame
    :param target_col: Target variable
    :param normality_map: Dictionary with normality info
    :param threshold: Correlation threshold above which features are considered redundant
    :param verbose: If True, prints removed features
    :return: Tuple (filtered_features, removed_features)
    """

    if len(features) <= 1:
        return features, []

    # Compute correlation with target for all features
    target_corr = {
        f: abs(_compute_correlation(df, f, target_col, normality_map)[0])
        for f in features
    }

    # Compute adaptive correlation matrix
    # Then restrict it only to the selected features
    corr_matrix_full, _, _ = adaptive_correlation_matrix(df[features])
    corr_matrix = corr_matrix_full.abs()

    # Upper triangle (avoid duplicate comparisons)
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_remove = set()

    # Iterate through pairs
    for col in upper.columns:
        for row in upper.index:

            if upper.loc[row, col] > threshold:

                # Decide which one to remove (keep the one more correlated with target)
                if target_corr[row] >= target_corr[col]:
                    to_remove.add(col)
                    removed_feature = col
                    kept_feature = row
                else:
                    to_remove.add(row)
                    removed_feature = row
                    kept_feature = col

                if verbose:
                    print(f"Removed {removed_feature} (kept {kept_feature})")

    kept_features = [f for f in features if f not in to_remove]

    return kept_features, list(to_remove)


# PCA
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


def top_correlations_per_target(
    targets: dict[str, list[str]],
    corr_df: pd.DataFrame,
    pval_df: pd.DataFrame,
    top_n: int = 10,
    sort_by: str = "abs_corr",
    alpha: float = 0.05,
    excel_path: str = "top_correlations.xlsx",
) -> pd.DataFrame:
    """
    Identify the top N variables most correlated with each target, restricted
    to statistically significant relationships (p-value < alpha).

    :param targets: Dict mapping each target (its keys are used as the
        list of targets) to the list of variables that must be excluded
        from that target's analysis
    :param corr_df: Correlation matrix (index and columns = variable names)
    :param pval_df: P-value matrix, with the same shape/index/columns as corr_df
    :param top_n: Number of most related variables to return per target
        (after filtering out non-significant ones)
    :param sort_by: "abs_corr" sorts by absolute correlation value (strongest
        relationship, positive or negative); "corr" sorts by the raw
        correlation value (highest to lowest)
    :param alpha: Significance threshold; only relationships with p-value < alpha
        are kept
    :param excel_path: Path of the Excel file to generate
    :return: DataFrame with columns Target, Variable, Correlation, P-value,
        Rank, also exported to Excel with one sheet per target
    """

    target_names = list(targets.keys())
    results = []

    for target in target_names:
        if target not in corr_df.columns:
            raise ValueError(f"Target '{target}' not found in corr_df columns.")

        # Variables to drop: the target itself + its excluded variables
        vars_to_drop = set(targets[target]) | {target}

        # Correlations and p-values of the target with all other variables
        target_corr = corr_df[target].drop(index=vars_to_drop, errors="ignore")
        target_pval = pval_df[target].drop(index=vars_to_drop, errors="ignore")

        # Combine everything into a single helper dataframe
        aux_df = pd.DataFrame({
            "Variable": target_corr.index,
            "Correlation": target_corr.values,
            "P-value": target_pval.reindex(target_corr.index).values,
        })

        # Keep only statistically significant correlations
        aux_df = aux_df[aux_df["P-value"] < alpha]

        # Sorting criterion
        if sort_by == "abs_corr":
            aux_df["_sort_key"] = aux_df["Correlation"].abs()
        elif sort_by == "corr":
            aux_df["_sort_key"] = aux_df["Correlation"]
        else:
            raise ValueError("sort_by must be 'abs_corr' or 'corr'.")

        aux_df = aux_df.sort_values("_sort_key", ascending=False).head(top_n)
        aux_df = aux_df.drop(columns="_sort_key")

        # Round correlation for final output
        aux_df["Correlation"] = aux_df["Correlation"].round(2)

        aux_df["P-value"] = aux_df["P-value"].apply(lambda p: "<0.001" if p < 0.001 else round(p, 3))

        aux_df.insert(0, "Target", target)

        results.append(aux_df)

    final_result = pd.concat(results, ignore_index=True)

    # Export to Excel
    if excel_path is not None:
        final_result.to_excel(excel_path, index=False)

        print(f"Excel exported to: {excel_path}")

    return final_result