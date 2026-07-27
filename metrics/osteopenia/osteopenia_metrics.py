from pathlib import Path

import pandas as pd

# Path to the osteopenia Excel file
OSTEOPENIA_EXCEL_PATH = Path(r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\tanita.xlsx")




def add_estimated_bone_mass() -> pd.DataFrame:
    """
    Read the Tanita Excel file and extract the estimated bone mass.

    ```
    :return: DataFrame containing the subject ID and estimated bone mass
    ```
    """

    # Read the Tanita data
    df = pd.read_excel(OSTEOPENIA_EXCEL_PATH)

    # Assume the first column contains the subject ID
    id_column = df.columns[0]

    # Keep only the ID and estimated bone mass columns
    df = df[[id_column, "Estimated bone mass kg"]].copy()

    return df