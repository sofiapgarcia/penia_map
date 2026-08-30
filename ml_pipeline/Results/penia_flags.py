"""
Penia group comparison analysis.

For each of the four Penia Map domains (sarcopenia, dynapenia, osteopenia,
powerpenia), participants are split into two groups (with penia / without
penia) using domain-specific criteria, and Estimated metabolic age is
compared between groups.

Sarcopenia, dynapenia and osteopenia use published clinical cut-offs.
Powerpenia has no published cut-off, so it is defined here as the lowest
peak_power quartile (bottom 25%), computed separately within each gender
group to avoid confounding by the well-established sex difference in
peak power.

Normality is checked per group (Shapiro-Wilk); an independent t-test is
used when both groups are normally distributed, otherwise a Mann-Whitney U
test is used. A Bonferroni-corrected alpha (0.05 / 4 domains = 0.0125) is
reported alongside the raw p-value, since four tests are run on the same
outcome variable.
"""

import pandas as pd
from scipy.stats import shapiro, ttest_ind, mannwhitneyu

# --------------------------------------------------------------------------
# CONFIG - adjust these to match the actual column names / encoding in the
# Excel file before running.
# --------------------------------------------------------------------------

EXCEL_PATH = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\final_dataframe.xlsx"
SHEET_NAME = 0  # sheet name or index

COL_METABOLIC_AGE = "Estimated metabolic age"
COL_ALMI = "ALMI"
COL_BMD = "Bone Mineral Density"
COL_GRIP_MAX = "hi_values_max"
COL_PEAK_POWER = "peak_power"
COL_GENDER = "Gender"

# Values used in COL_GENDER to identify each sex. Adjust to match the actual
# encoding in the file (e.g. 1/0, "M"/"F", "Male"/"Female").
MALE_VALUE = 1
FEMALE_VALUE = 0

BONFERRONI_ALPHA = 0.05 / 4

# --------------------------------------------------------------------------
# Thresholds (Sarcopenia, Osteopenia, Dynapenia)
# --------------------------------------------------------------------------

SARCOPENIA_CUTOFF = {"M": 8.87, "F": 6.42}  # ALMI (kg/m2), BIA-based cutoff, below cutoff = sarcopenia

OSTEOPENIA_RANGE = {  # BMD (g/cm2), within interval (inclusive) = osteopenia
    "M": (0.946, 1.096),
    "F": (0.9035, 1.025),
}

DYNAPENIA_CUTOFF = {"M": 30, "F": 20}  # hi_values_max (kg), below cutoff = dynapenia

POWERPENIA_QUARTILE = 0.25  # bottom quartile of peak_power, within each sex


def load_data(path: str, sheet: str = 0) -> pd.DataFrame:
    """
    Load the dataset from an Excel file.

    :param path: path to the .xlsx file
    :param sheet: sheet name or index to read
    :return: loaded dataframe
    """
    df = pd.read_excel(path, sheet_name=sheet)
    print(f"Loaded {len(df)} rows from '{path}'.")
    print(f"Unique values in '{COL_GENDER}': {df[COL_GENDER].unique().tolist()}")
    return df


def add_sarcopenia_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a binary sarcopenia flag based on sex-specific ALMI cut-offs.

    :param df: input dataframe, must contain COL_ALMI and COL_GENDER
    :return: dataframe with a new 'sarcopenia_flag' column
    """
    cutoff = df[COL_GENDER].map({MALE_VALUE: SARCOPENIA_CUTOFF["M"],
                                  FEMALE_VALUE: SARCOPENIA_CUTOFF["F"]})
    df["sarcopenia_flag"] = (df[COL_ALMI] < cutoff).astype(int)
    return df


def add_osteopenia_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a binary osteopenia flag based on sex-specific BMD reference upper
    bounds. Any BMD below the upper bound of the reference interval is
    flagged (this captures both osteopenia and osteoporosis, which sit on
    the same low-bone-density spectrum, osteoporosis being the more severe
    end below the interval's lower bound).

    :param df: input dataframe, must contain COL_BMD and COL_GENDER
    :return: dataframe with a new 'osteopenia_flag' column
    """
    upper = df[COL_GENDER].map({MALE_VALUE: OSTEOPENIA_RANGE["M"][1],
                                 FEMALE_VALUE: OSTEOPENIA_RANGE["F"][1]})
    df["osteopenia_flag"] = (df[COL_BMD] <= upper).astype(int)
    return df


def add_dynapenia_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a binary dynapenia flag based on sex-specific grip strength cut-offs.

    :param df: input dataframe, must contain COL_GRIP_MAX and COL_GENDER
    :return: dataframe with a new 'dynapenia_flag' column
    """
    cutoff = df[COL_GENDER].map({MALE_VALUE: DYNAPENIA_CUTOFF["M"],
                                  FEMALE_VALUE: DYNAPENIA_CUTOFF["F"]})
    df["dynapenia_flag"] = (df[COL_GRIP_MAX] < cutoff).astype(int)
    return df


def add_powerpenia_flag(df: pd.DataFrame, quantile: float = POWERPENIA_QUARTILE) -> pd.DataFrame:
    """
    Add a binary powerpenia flag: 1 if peak_power falls at or below the
    given quantile (default: bottom quartile), computed separately within
    each sex group.

    :param df: input dataframe, must contain COL_PEAK_POWER and COL_GENDER
    :param quantile: quantile threshold (default 0.25, i.e. bottom 25%)
    :return: dataframe with a new 'powerpenia_flag' column
    """
    thresholds = df.groupby(COL_GENDER)[COL_PEAK_POWER].transform(
        lambda x: x.quantile(quantile)
    )
    df["powerpenia_flag"] = (df[COL_PEAK_POWER] <= thresholds).astype(int)
    return df


def compare_groups(df: pd.DataFrame, flag_col: str, outcome_col: str = COL_METABOLIC_AGE) -> dict:
    """
    Compare the outcome variable between the 'with penia' and 'without
    penia' groups defined by a binary flag column, choosing the test based
    on within-group normality (Shapiro-Wilk).

    :param df: dataframe containing the flag and outcome columns
    :param flag_col: name of the binary flag column (0/1)
    :param outcome_col: name of the outcome variable to compare
    :return: dict with test results and group sizes
    """
    with_penia = df.loc[df[flag_col] == 1, outcome_col].dropna()
    without_penia = df.loc[df[flag_col] == 0, outcome_col].dropna()

    result = {
        "domain": flag_col,
        "n_with": len(with_penia),
        "n_without": len(without_penia),
    }

    # Shapiro-Wilk requires at least 3 observations per group
    if len(with_penia) < 3 or len(without_penia) < 3:
        result.update({"test": "not enough data", "statistic": None, "p_value": None})
        return result

    _, p_with = shapiro(with_penia)
    _, p_without = shapiro(without_penia)

    if p_with > 0.05 and p_without > 0.05:
        stat, p = ttest_ind(with_penia, without_penia, equal_var=False)
        test_name = "Welch's t-test"
    else:
        stat, p = mannwhitneyu(with_penia, without_penia)
        test_name = "Mann-Whitney U"

    result.update({
        "test": test_name,
        "statistic": stat,
        "p_value": p,
        "significant_bonferroni": p < BONFERRONI_ALPHA if p is not None else None,
    })
    return result


def main():
    df = load_data(EXCEL_PATH, SHEET_NAME)

    df = add_sarcopenia_flag(df)
    df = add_dynapenia_flag(df)
    df = add_osteopenia_flag(df)
    df = add_powerpenia_flag(df)

    flags = ["sarcopenia_flag", "dynapenia_flag", "osteopenia_flag", "powerpenia_flag"]

    print("\nGroup sizes per domain:")
    for flag in flags:
        print(f"  {flag}: {df[flag].value_counts().to_dict()}")

    print(f"\nComparing {COL_METABOLIC_AGE} between 'with penia' vs 'without penia' groups")
    print(f"(Bonferroni-corrected alpha for 4 tests: {BONFERRONI_ALPHA:.4f})\n")

    results = [compare_groups(df, flag) for flag in flags]
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    output_path = "penia_group_comparison_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to '{output_path}'.")


if __name__ == "__main__":
    main()