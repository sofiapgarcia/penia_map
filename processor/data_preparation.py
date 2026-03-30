"""
Basic data quality utilities.

Available Functions
-------------------
[Public]
analyze_excel_file(...):

[Private]
_detect_outliers_iqr(...):
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder


# ------------------------------------------------------------------------------------------------------------------- #
# file specific constants
# ------------------------------------------------------------------------------------------------------------------- #

EXCLUDE_FROM_OUTLIER_ANALYSIS = [
    "Gender",
    "IPAQ",
    "HHIE-S (Classification)",
    "Dominat Hand",
    "Marital Status",
    "Education Level",
    "Stimulus",
]


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #


def analyze_excel_file(
    file_path: str,
    non_numeric_columns: list = EXCLUDE_FROM_OUTLIER_ANALYSIS,
) -> pd.DataFrame:
    """
    Load an Excel file and perform basic data checks.

    The function:
    - Loads the Excel file
    - Warns if missing values exist
    - Warns if duplicate rows exist
    - Detects outliers using the IQR method
    - Prints affected columns
    - Returns the DataFrame

    :param file_path: Path to the Excel file
    :param non_numeric_columns: Columns excluded from outlier analysis
    :return: Loaded DataFrame
    """

    df = pd.read_excel(file_path)

    print(f"File loaded: {df.shape[0]} rows, {df.shape[1]} columns")


    # ---- Missing values ----
    total_missing = df.isna().sum().sum()

    if total_missing > 0:
        print(f"WARNING: Missing values detected ({total_missing})")
    else:
        print("No missing values detected")

    # ---- Duplicates ----
    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        print(f"WARNING: Duplicate rows detected ({duplicate_count})")
    else:
        print("No duplicate rows detected")

    # ---- Outliers ----
    outlier_columns = _detect_outliers_iqr(
        df,
        non_numeric_columns,
    )


    if outlier_columns:
        print(
            f"Outliers detected in {len(outlier_columns)} columns:",
            outlier_columns,
        )
        plot_outlier_details(df, outlier_columns)

        # ---- Outlier IDs ----
        outlier_ids = identify_outlier_ids(df, non_numeric_columns)
        print("\nIDs with more outliers:")
        print(outlier_ids.to_string(index=False))

    else:
        print("No outliers detected")

    return df


def plot_outlier_details(df: pd.DataFrame, columns: list):
    """
    Plot detailed distribution for columns with outliers.

    Shows:
    - Histogram with KDE
    - Boxplot
    - IQR bounds
    - Number of outliers
    - Exact outlier values

    :param df: Input DataFrame
    :param columns: Columns to visualize
    """

    for col in columns:

        data = df[col].dropna()

        # ---- IQR calculation ----
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = data[(data < lower) | (data > upper)]

        # ---- Plot ----
        plt.figure(figsize=(12, 4))

        # Histogram
        plt.subplot(1, 2, 1)
        sns.histplot(data, kde=True)

        plt.axvline(lower, linestyle="--")
        plt.axvline(upper, linestyle="--")

        plt.title(
            f"{col} — Distribution\n"
            f"Outliers: {len(outliers)}"
        )

        # Boxplot
        plt.subplot(1, 2, 2)
        sns.boxplot(x=data)

        plt.title(f"{col} — Boxplot")

        plt.tight_layout()
        plt.show(block=False)

        # ---- Print values ----
        if len(outliers) > 0:
            print(f"{col} outlier values:")
            print(sorted(outliers.values))
            print("-" * 50)

    plt.show()

def identify_outlier_ids(df: pd.DataFrame, non_numeric_columns: list = EXCLUDE_FROM_OUTLIER_ANALYSIS, id_column: str = "ID") -> pd.DataFrame:
    """
    Identify which IDs have the most columns with outlier values.

    :param df: Input DataFrame
    :param non_numeric_columns: Columns to exclude from outlier analysis
    :param id_column: Name of the ID column
    :return: DataFrame with ID and outlier column count, sorted descending
    """

    numeric_cols = df.select_dtypes(include="number").columns
    columns_to_check = [col for col in numeric_cols if col not in non_numeric_columns]

    outlier_mask = pd.DataFrame(False, index=df.index, columns=columns_to_check)

    for col in columns_to_check:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outlier_mask[col] = (df[col] < lower) | (df[col] > upper)

    result = pd.DataFrame({
        id_column: df[id_column],
        "outlier_count": outlier_mask.sum(axis=1),
        "outlier_columns": outlier_mask.apply(lambda row: list(row[row].index), axis=1)
    })

    result = result[result["outlier_count"] > 0].sort_values("outlier_count", ascending=False)

    return result



def encode_categorical_columns(
    df: pd.DataFrame,
    columns: list = EXCLUDE_FROM_OUTLIER_ANALYSIS,
) -> pd.DataFrame:
    """
    One-hot encode specified categorical columns using sklearn's OneHotEncoder,
    avoiding the dummy variable trap by dropping the first category.

    :param df: Input DataFrame
    :param columns: List of categorical columns to encode
    :return: DataFrame with encoded columns
    """

    df_encoded = df.copy()

    # Check for missing columns
    missing_cols = [col for col in columns if col not in df_encoded.columns]
    if missing_cols:
        print(f"WARNING: Columns not found and skipped: {missing_cols}")

    valid_cols = [col for col in columns if col in df_encoded.columns]

    if not valid_cols:
        print("No valid columns to encode.")
        return df_encoded

    encoder = OneHotEncoder(
        drop="first",        # 🔥 avoids dummy variable trap
        sparse_output=False,
        handle_unknown="ignore"
    )

    encoded_array = encoder.fit_transform(df_encoded[valid_cols])

    encoded_cols = encoder.get_feature_names_out(valid_cols)

    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoded_cols,
        index=df_encoded.index
    )

    # Drop original categorical columns
    df_encoded = df_encoded.drop(columns=valid_cols)

    # Concatenate encoded columns
    df_encoded = pd.concat([df_encoded, encoded_df], axis=1)

    print(f"One-hot encoding complete. Columns encoded: {valid_cols}")

    return df_encoded

# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #


def _detect_outliers_iqr(
    df: pd.DataFrame,
    non_numeric_columns: list,
) -> list:
    """
    Detect columns containing outliers using the IQR method.

    :param df: Input DataFrame
    :param non_numeric_columns: Columns to exclude
    :return: List of column names with detected outliers
    """

    # Ensure ID is always excluded
    columns_to_exclude = non_numeric_columns + ["ID"]

    numeric_cols = df.select_dtypes(include="number").columns
    columns_to_check = [col for col in numeric_cols if col not in columns_to_exclude]

    outlier_columns = []

    for col in columns_to_check:

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        if ((df[col] < lower) | (df[col] > upper)).any():
            outlier_columns.append(col)

    return outlier_columns