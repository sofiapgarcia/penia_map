from ml_pipeline.preprocessing.feature_reduction import adaptive_correlation_matrix
from constants import PENIA_MAP_GROUPS

import pandas as pd


# ============================================================
# Configuration
# ============================================================

FILE_PATH = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\final_dataframe.xlsx"

TARGET = "Age (y)"
#TARGET = "Estimated metabolic age"
#TARGET = "Bone Mineral Density"
#TARGET = "peak_power"

TOP_N = 10

ALPHA = 0.05


# ============================================================
# Read data
# ============================================================

df = pd.read_excel(FILE_PATH)


# ============================================================
# Compute adaptive correlation matrix
# ============================================================

corr_matrix, pval_matrix, method_matrix = adaptive_correlation_matrix(
    df,
    alpha=ALPHA
)


# ============================================================
# Find top significant correlations for each domain
# ============================================================

for domain, features in PENIA_MAP_GROUPS.items():

    # Keep only features that exist in the dataframe
    available_features = [
        feature
        for feature in features
        if feature in df.columns
    ]

    if not available_features:
        print(f"\n{domain}: no available features.")
        continue

    # Get correlations and p-values with target
    correlations = corr_matrix.loc[available_features, TARGET]
    p_values = pval_matrix.loc[available_features, TARGET]

    # ========================================================
    # Separate significant and non-significant features
    # ========================================================

    significant_features = p_values[p_values < ALPHA].index
    non_significant_features = p_values[p_values >= ALPHA].index

    # ========================================================
    # Print non-significant features
    # ========================================================

    print("\n" + "=" * 80)
    print(f"{domain} - NON-SIGNIFICANT FEATURES")
    print("=" * 80)

    if len(non_significant_features) == 0:

        print("All features were statistically significant.")

    else:

        for feature in non_significant_features:

            correlation = corr_matrix.loc[feature, TARGET]
            p_value = pval_matrix.loc[feature, TARGET]

            print(
                f"{feature}: "
                f"r = {correlation:.3f}, "
                f"p = {p_value:.4g}"
            )

    # ========================================================
    # Filter significant correlations
    # ========================================================

    if len(significant_features) == 0:

        print("\n" + "=" * 80)
        print(f"{domain} - TOP {TOP_N} SIGNIFICANT FEATURES")
        print("=" * 80)
        print("No statistically significant correlations.")
        continue

    significant_correlations = correlations.loc[significant_features]

    # ========================================================
    # Sort significant correlations by absolute correlation
    # ========================================================

    significant_correlations = significant_correlations.reindex(
        significant_correlations.abs()
        .sort_values(ascending=False)
        .index
    )

    # ========================================================
    # Print top N significant correlations
    # ========================================================

    print("\n" + "=" * 80)
    print(f"{domain} - TOP {TOP_N} SIGNIFICANT FEATURES")
    print(f"Target: {TARGET}")
    print("=" * 80)

    for feature in significant_correlations.head(TOP_N).index:

        correlation = corr_matrix.loc[feature, TARGET]
        p_value = pval_matrix.loc[feature, TARGET]
        method = method_matrix.loc[feature, TARGET]

        print(
            f"{feature}: "
            f"r = {correlation:.3f}, "
            f"p = {p_value:.4g}, "
            f"method = {method}"
        )