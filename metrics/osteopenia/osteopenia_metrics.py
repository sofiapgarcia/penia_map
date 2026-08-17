from pathlib import Path

import pandas as pd

# Path to the Tanita Excel file
OSTEOPENIA_EXCEL_PATH = Path(r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\tanita.xlsx")


def add_bone_variables() -> pd.DataFrame:
    """
    Read the Tanita Excel file and calculate estimated bone mass and
    Bone Mineral Density (BMD).

    :return: DataFrame containing the subject ID, estimated bone mass,
             and Bone Mineral Density
    """

    # Read the Tanita data
    df = pd.read_excel(OSTEOPENIA_EXCEL_PATH)

    # Assume the first column contains the subject ID
    id_column = df.columns[0]

    # Calculate Bone Mineral Density
    df["Bone Mineral Density"] = (
        1.46378
        - 0.00603 * df["Height(cm)"]
        + 0.30443 * df["Estimated bone mass kg"]
        - 0.00112 * df["Leg fat (left) %"]
    )

    # Keep only the ID and bone-related variables
    df = df[
        [
            id_column,
            "Estimated bone mass kg",
            "Bone Mineral Density",
        ]
    ].copy()

    return df