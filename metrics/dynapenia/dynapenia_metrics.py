from pathlib import Path

import pandas as pd

# Path to the input Excel file
EXCEL_PATH = Path(r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\handgrip_isometric.xlsx")


def add_max_column_from_excel() -> pd.DataFrame:
    """
    Read an Excel file and create a new column containing the maximum value
    across the three data columns.

    ```
    :return: DataFrame with a new column containing the row-wise maximum
             of the three data columns
    ```
    """

    # Read the Excel file into a DataFrame
    df = pd.read_excel(EXCEL_PATH)

    # Assume the first column is the subject ID and the next three columns
    # are the values used to compute the maximum
    value_columns = df.columns[1:3]

    # Compute the row-wise maximum across the three value columns
    df["HI_max"] = df[value_columns].max(axis=1)

    return df