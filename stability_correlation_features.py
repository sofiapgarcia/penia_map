from ml_pipeline.preprocessing.linear_regression_feature_reduction import adaptive_correlation_matrix , top_correlations_per_target
from ml_pipeline.preprocessing.linear_regression_feature_reduction import run_for_multiple_targets , _results_to_dataframe
from constants import RANDOM_SEED , best_models_all_variables , best_models_penia_map
import pandas as pd

filepath = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\final_dataframe.xlsx"

# read the data
df = pd.read_excel(filepath)

# remove participant column ID
df_features = df.drop(columns=["ID"], errors="ignore")

cor_matrix , corr_pvalues , corr_coef = adaptive_correlation_matrix(df_features)

targets = {"Estimated metabolic age": ["Age (y)"],
"peak_power": ["velocity_at_peak_power", "force_at_peak_power", "peak_power_mass", "peak_power_height_squared", "max_force", "max_velocity","rep_pp_hd_peak_rfd","mean_hd_peak_rfd"],
"Bone Mineral Density": ["Estimated bone mass kg"],
           "Age (y)":["Estimated metabolic age"]
}

import pandas as pd
from collections import Counter, defaultdict

import pandas as pd
from collections import Counter, defaultdict

def analyze_feature_stability(
    df_features,
    targets,
    strat_col="Age (y)",
    n_iterations=100,
    test_size=0.15,
    top_n=10,
    sort_by="abs_corr",
    alpha=0.05,
    base_seed=0
):
    """
    Repeatedly splits the dataset into train/test (stratifying by strat_col),
    recomputes the correlation matrix on df_train only, and counts how many
    times each variable appears in the top_n most correlated variables for
    each target across iterations.

    :param df_features: Full feature dataframe (target columns included)
    :param targets: Dict mapping each target to the list of variables to
        exclude from its own analysis
    :param strat_col: Column used to stratify the train/test split in every
        iteration
    :param n_iterations: Number of random train/test splits to perform
    :param test_size: Proportion of data held out as test set in each split
    :param top_n: Number of top correlated variables to keep per iteration
    :param sort_by: Passed through to top_correlations_per_target
        ("abs_corr" or "corr")
    :param alpha: Significance threshold passed through to
        top_correlations_per_target
    :param base_seed: Base random seed; iteration i uses base_seed + i
    :return: Dict {target: stability_dataframe}, where stability_dataframe has
        columns [variable, n_appearances, pct_appearances, avg_correlation]
        sorted by n_appearances descending
    """
    appearance_counts = {target: Counter() for target in targets}
    corr_sums = {target: defaultdict(float) for target in targets}
    valid_iterations = {target: 0 for target in targets}

    for i in range(n_iterations):
        seed = base_seed + i

        df_train, df_test = split_train_test_df(
            df_features,
            test_size=test_size,
            random_state=seed,
            target_col=strat_col
        )

        cor_matrix_i, corr_pvalues_i, _ = adaptive_correlation_matrix(df_train)

        result_i = top_correlations_per_target(
            targets=targets,
            corr_df=cor_matrix_i,
            pval_df=corr_pvalues_i,
            top_n=top_n,
            sort_by=sort_by,
            alpha=alpha,
            excel_path=None
        )

        for target_name in targets:
            target_rows = result_i[result_i["Target"] == target_name]
            if target_rows.empty:
                continue

            valid_iterations[target_name] += 1

            for _, row in target_rows.iterrows():
                var = row["Variable"]
                appearance_counts[target_name][var] += 1
                corr_sums[target_name][var] += row["Correlation"]

    stability_results = {}
    for target_name in targets:
        rows = []
        for var, count in appearance_counts[target_name].most_common():
            rows.append({
                "variable": var,
                "n_appearances": count,
                "pct_appearances": round(100 * count / valid_iterations[target_name], 1) if valid_iterations[target_name] else 0,
                "avg_correlation": round(corr_sums[target_name][var] / count, 4)
            })

        stability_df = pd.DataFrame(rows).sort_values("n_appearances", ascending=False).reset_index(drop=True)
        stability_results[target_name] = stability_df

    return stability_results

stability_results = analyze_feature_stability(
    df_features=df_features,
    targets=targets,
    strat_col="Age (y)",
    n_iterations=40,
    test_size=0.15,
    top_n=10
)

for target, df_summary in stability_results.items():
    print(f"\n=== {target} ===")
    print(df_summary.head(15))

output_path = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\results\feature_stability.xlsx"

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    for target_name, stability_df in stability_results.items():
        sheet_name = target_name[:31].replace("/", "-").replace("\\", "-")
        stability_df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"Excel saved in: {output_path}")