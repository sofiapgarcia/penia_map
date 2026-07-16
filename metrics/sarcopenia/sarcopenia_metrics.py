from pathlib import Path

import pandas as pd

# Path to the sarcopenia Excel file
SARCOPENIA_EXCEL_PATH = Path(r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\tanita.xlsx")

SARCOPENIA_METRICS = [
    "Global muscle kg",
    "Arm muscle (right) kg",
    "Arm muscle (left) kg",
    "Leg muscle (right) kg",
    "Leg muscle (left) kg",
    "Torso muscle kg",
]


def add_almi() -> pd.DataFrame:
    """
    Read the sarcopenia Excel file, extract the required muscle metrics,
    and compute the Appendicular Lean Mass Index (ALMI).

    ```
    :return: DataFrame containing the selected sarcopenia metrics and the
             ALMI column
    ```
    """

    # Read the sarcopenia Excel file
    df = pd.read_excel(SARCOPENIA_EXCEL_PATH)

    # Assume the first column contains the subject ID
    id_column = df.columns[0]

    # Keep only the ID column and the required sarcopenia metrics
    df = df[[id_column, "Height(cm)"] + SARCOPENIA_METRICS].copy()

    # Convert height from centimeters to meters
    height_m = df["Height(cm)"] / 100

    # Compute ALMI = global muscle mass / height²
    df["ALMI"] = df["Global muscle kg"] / (height_m**2)

    # Remove the temporary height column
    df = df.drop(columns=["Height(cm)"])

    return df

