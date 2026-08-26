"""
file to test if the correlation are higher with the proxies than with the chronological age
"""
from scipy.stats import wilcoxon
import pandas as pd
import numpy as np
from ml_pipeline.preprocessing.feature_reduction import adaptive_correlation_matrix


def compare_correlations_with_age(
    corr_coef: pd.DataFrame,
    targets: dict[str, list[str]],
    age_column: str = "Age (y)",
) -> pd.DataFrame:
    """
    Compare the absolute Spearman correlations of each target
    against the absolute correlations with chronological age.

    Variables listed in targets[target] are excluded for that target.

    For each target:

        difference = |rho_target| - |rho_age|

    A one-sided Wilcoxon signed-rank test is performed:

        H0: median difference <= 0
        H1: median difference > 0

    Returns one row per target.
    """

    results = []

    for target, excluded_variables in targets.items():

        # Age is the reference, not a biological-age proxy
        if target == age_column:
            continue

        # Variables that should not be considered for this target
        excluded = set(excluded_variables)
        excluded.update([target, age_column])

        target_abs_corr = []
        age_abs_corr = []
        differences = []

        for var in corr_coef.index:

            # Exclude variables according to the target-specific dictionary
            if var in excluded:
                continue

            # Need both correlations
            if target not in corr_coef.columns:
                continue

            if age_column not in corr_coef.columns:
                continue

            rho_target = corr_coef.loc[var, target]
            rho_age = corr_coef.loc[var, age_column]

            if pd.isna(rho_target) or pd.isna(rho_age):
                continue

            abs_target = abs(rho_target)
            abs_age = abs(rho_age)

            target_abs_corr.append(abs_target)
            age_abs_corr.append(abs_age)
            differences.append(abs_target - abs_age)

        differences = np.asarray(differences)

        if len(differences) == 0:
            continue

        # Wilcoxon signed-rank test
        statistic, p_value = wilcoxon(
            differences,
            alternative="greater"
        )

        results.append({
            "Target": target,
            "N variables": len(differences),
            "Median |rho| target": np.median(target_abs_corr),
            "Median |rho| age": np.median(age_abs_corr),
            "Median difference": np.median(differences),
            "Wilcoxon W": statistic,
            "P-value": p_value,
        })

    return pd.DataFrame(results)



filepath = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\final_dataframe.xlsx"

# read the data
df = pd.read_excel(filepath)

# remove participant column ID
df_features = df.drop(columns=["ID","Torso muscle kg"], errors="ignore")


cor_matrix , corr_pvalues , corr_coef = adaptive_correlation_matrix(df_features)


targets = {"Estimated metabolic age": ["Age (y)"],
"peak_power": ["velocity_at_peak_power", "force_at_peak_power", "peak_power_mass", "peak_power_height_squared", "max_force", "max_velocity","rep_pp_hd_peak_rfd","mean_hd_peak_rfd"],
"Bone Mineral Density": ["Estimated bone mass kg"],
           "Age (y)":["Estimated metabolic age"]
}
comparison_results = compare_correlations_with_age(
    corr_coef=cor_matrix,
    targets=targets,
    age_column="Age (y)"
)

print(comparison_results.to_string(index=False))