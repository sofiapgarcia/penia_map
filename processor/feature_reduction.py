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


# ------------------------------------------------------------------------------------------------------------------- #
# file specific constants
# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def spearman_correlation_with_target(df, target_col = "Estimated metabolic age", top_n=None, ascending=False):
    """
    Compute Spearman correlation between all numeric features and a target column.

    :param df: Input DataFrame
    :param target_col: Name of the target column (e.g., "Estimated metabolic age")
    :param top_n: Number of top features to return (optional)
    :param ascending: Sort correlations in ascending order if True
    :return: DataFrame with features and their Spearman correlation with the target, sorted
    """

    # Check if target column exists
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in DataFrame.")

    # Select only numeric columns
    numeric_df = df.select_dtypes(include='number')

    # Compute Spearman correlation matrix and extract target correlations
    correlations = numeric_df.corr(method='spearman')[target_col]

    # Remove self-correlation (target vs target)
    correlations = correlations.drop(target_col)

    # Sort correlations
    correlations = correlations.sort_values(ascending=ascending)

    # Convert to DataFrame
    correlations_df = correlations.to_frame(name='Spearman Correlation')

    # Return only top N features if specified
    if top_n is not None:
        correlations_df = correlations_df.head(top_n)

    return correlations_df