"""
Outlier detection using the Interquartile Range (IQR) method.

For each numeric column, values below (Q1 - 1.5 * IQR) or above
(Q3 + 1.5 * IQR) are considered outliers and replaced with NaN.
A small textual summary of what was found is also generated.

Variables that are computed from other variables (ratios, normalized
versions, abs versions, etc.) are then propagated to NaN wherever their
origin variable(s) were flagged as outliers, using DEPENDENCY_MAP.
"""

import numpy as np
import pandas as pd

from dependency_map import DEPENDENCY_MAP


def detect_outliers_iqr(df, non_numeric_columns=None, factor=1.5):
    """
    Detect outliers in numeric columns using the IQR method and
    replace them with NaN.

    :param df: Input DataFrame
    :param non_numeric_columns: Columns to exclude from outlier analysis
    :param factor: Multiplier applied to the IQR to define the bounds (default 1.5)
    :return: Tuple (cleaned DataFrame, report dict with details per column)
    """
    non_numeric_columns = non_numeric_columns or []
    df_clean = df.copy()
    report = {}

    numeric_columns = [
        col for col in df_clean.columns
        if col not in non_numeric_columns and pd.api.types.is_numeric_dtype(df_clean[col])
    ]

    for col in numeric_columns:
        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr

        outlier_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
        outlier_count = int(outlier_mask.sum())

        if outlier_count > 0:
            df_clean.loc[outlier_mask, col] = np.nan

        report[col] = {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outlier_count": outlier_count,
        }

    return df_clean, report


def propagate_nan_dependencies(df, dependency_map):
    """
    Propagate NaN values from origin variables to variables derived from them.

    Runs in a loop until no further changes occur, so that dependency chains
    of more than one level (A -> B -> C) are fully resolved without needing
    to sort the map topologically by hand.

    :param df: DataFrame where outliers have already been replaced with NaN
        in the origin columns
    :param dependency_map: dict {derived_variable: [origin_variable, ...]}
    :return: Tuple (DataFrame with NaN propagated, report dict with the
        number of newly-propagated NaN values per derived column)
    """
    df = df.copy()
    propagation_report = {}
    changed = True

    while changed:
        changed = False
        for derived, origins in dependency_map.items():
            if derived not in df.columns:
                continue

            origins_present = [o for o in origins if o in df.columns]
            missing_origins = [o for o in origins if o not in df.columns]
            if missing_origins:
                print(f"Warning: origin column(s) {missing_origins} for '{derived}' not found in DataFrame.")
            if not origins_present:
                continue

            mask = df[origins_present].isna().any(axis=1)
            newly_nan = mask & df[derived].notna()
            newly_nan_count = int(newly_nan.sum())

            if newly_nan_count > 0:
                df.loc[mask, derived] = np.nan
                propagation_report[derived] = propagation_report.get(derived, 0) + newly_nan_count
                changed = True

    return df, propagation_report


def describe_outliers(report):
    """
    Build a short human-readable description of the outliers found.

    :param report: Report dict produced by detect_outliers_iqr
    :return: String summary describing the outliers found per column
    """
    columns_with_outliers = {col: info for col, info in report.items() if info["outlier_count"] > 0}

    if not columns_with_outliers:
        return "No outliers were found in any numeric column."

    total_outliers = sum(info["outlier_count"] for info in columns_with_outliers.values())
    lines = [f"Found {total_outliers} outlier value(s) across {len(columns_with_outliers)} column(s):"]

    for col, info in columns_with_outliers.items():
        lines.append(
            f"  - '{col}': {info['outlier_count']} outlier(s) "
            f"(valid range: [{info['lower_bound']:.2f}, {info['upper_bound']:.2f}])"
        )

    return "\n".join(lines)


def describe_propagation(propagation_report):
    """
    Build a short human-readable description of the NaN values propagated
    to derived variables.

    :param propagation_report: Report dict produced by propagate_nan_dependencies
    :return: String summary describing the propagation per column
    """
    if not propagation_report:
        return "No NaN values were propagated to derived columns."

    total_propagated = sum(propagation_report.values())
    lines = [f"Propagated NaN to {total_propagated} value(s) across {len(propagation_report)} derived column(s):"]

    for col, count in propagation_report.items():
        lines.append(f"  - '{col}': {count} value(s) set to NaN")

    return "\n".join(lines)


def describe_total_nan(df, non_numeric_columns=None):
    """
    Build a short human-readable summary of the total NaN values currently
    present in the DataFrame.

    :param df: DataFrame to inspect (e.g. after outlier removal and propagation)
    :param non_numeric_columns: Columns to exclude from the count
    :return: String summary with the total NaN count and count per column
    """
    non_numeric_columns = non_numeric_columns or []
    columns_to_check = [col for col in df.columns if col not in non_numeric_columns]

    nan_per_column = df[columns_to_check].isna().sum()
    nan_per_column = nan_per_column[nan_per_column > 0].sort_values(ascending=False)
    total_nan = int(nan_per_column.sum())

    if total_nan == 0:
        return "No NaN values remain in the DataFrame."

    lines = [f"{total_nan} NaN value(s) remain across {len(nan_per_column)} column(s):"]
    for col, count in nan_per_column.items():
        lines.append(f"  - '{col}': {int(count)} NaN")

    return "\n".join(lines)


def get_outlier_counts_per_id(df, non_numeric_columns, id_column):
    """
    Identify which IDs have the most columns with outlier values.

    :param df: Input DataFrame
    :param non_numeric_columns: Columns to exclude from outlier analysis
    :param id_column: Name of the ID column
    :return: DataFrame with ID and outlier column count, sorted descending
    """
    _, report = detect_outliers_iqr(df, non_numeric_columns=non_numeric_columns)

    # Recompute outlier masks per column to know which rows (IDs) are affected.
    # Build all columns in a dict first and concat once, to avoid the
    # "highly fragmented DataFrame" performance warning from inserting
    # columns one at a time.
    numeric_columns = list(report.keys())
    flags_dict = {
        col: (df[col] < report[col]["lower_bound"]) | (df[col] > report[col]["upper_bound"])
        for col in numeric_columns
    }
    outlier_flags = pd.concat(flags_dict, axis=1)

    counts_per_id = pd.DataFrame({
        id_column: df[id_column],
        "outlier_column_count": outlier_flags.sum(axis=1),
    })

    counts_per_id = counts_per_id.sort_values(by="outlier_column_count", ascending=False).reset_index(drop=True)

    return counts_per_id


if __name__ == "__main__":
    # ---- Configuration ----
    input_path = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\final_dataframe.xlsx"      # path to the Excel file to read
    sheet_name = 0                  # sheet index or name
    output_path = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\final_dataframe_without_outliers.xlsx"  # path to save the cleaned DataFrame
    non_numeric_columns = ["ID","Activity level"]  # columns to exclude from outlier analysis
    id_column = "ID"                # name of the ID column

    # ---- Read ----
    df = pd.read_excel(input_path, sheet_name=sheet_name)

    # ---- Detect and remove outliers ----
    cleaned_df, outlier_report = detect_outliers_iqr(df, non_numeric_columns=non_numeric_columns)

    # ---- Propagate NaN to derived variables ----
    cleaned_df, propagation_report = propagate_nan_dependencies(cleaned_df, DEPENDENCY_MAP)

    # ---- Description of what was found ----
    print(describe_outliers(outlier_report))
    print()
    print(describe_propagation(propagation_report))
    print()

    # ---- IDs with the most columns affected ----
    print(get_outlier_counts_per_id(df, non_numeric_columns=non_numeric_columns, id_column=id_column))

    print("number of columns: ", len(df.columns)-1)

    # ---- Final NaN count (outliers + propagation) ----
    print()
    print(describe_total_nan(cleaned_df, non_numeric_columns=non_numeric_columns))

    # ---- Save cleaned DataFrame ----
    cleaned_df.to_excel(output_path, index=False)
    print(f"\nCleaned data saved to '{output_path}'")