"""
Functions for computing adaptive correlation matrices and pairwise correlation statistics.

Available Functions
-------------------
[Public]
adaptive_correlation_matrix(...): Computes correlation coefficients, p-values and correlation methods for all pairs of numeric
variables, selecting Pearson or Spearman correlation based on normality.

[Private]
_compute_normality_map(...): Evaluates the normality of numeric variables using the Shapiro-Wilk test and maps each variable to
its normality status.
_compute_correlation(...): Computes the correlation coefficient and p-value for a pair of variables using Pearson or Spearman
correlation according to their normality.
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import pandas as pd
from scipy import stats

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

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

    # Correlation of a variable with itself (diagonal of the matrix)
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