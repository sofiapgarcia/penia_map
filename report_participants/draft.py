
import pandas as pd
from pathlib import Path


def merge_participant_data(path_1, path_2, output_path):
    """
    Merge two participant datasets containing different participant IDs.

    :param path_1: Path to the first Excel file.
    :param path_2: Path to the second Excel file.
    :param output_path: Path where the merged dataset will be saved.
    """

    # Load datasets
    df_1 = pd.read_excel(path_1)
    df_2 = pd.read_excel(path_2)

    # Initial information
    print("\n--- ORIGINAL DATASETS ---")
    print(f"Dataset 1: {len(df_1)} participants")
    print(f"Dataset 2: {len(df_2)} participants")

    print(f"Dataset 1 IDs: {df_1['ID'].tolist()}")
    print(f"Dataset 2 IDs: {df_2['ID'].tolist()}")

    # Check columns
    if set(df_1.columns) != set(df_2.columns):
        raise ValueError("The datasets do not have the same columns.")

    # Check for overlapping IDs
    common_ids = set(df_1["ID"]) & set(df_2["ID"])

    if common_ids:
        raise ValueError(f"Duplicate participant IDs found: {common_ids}")

    # Merge datasets (append rows)
    df_merged = pd.concat([df_1, df_2], ignore_index=True)

    # Verification
    print("\n--- MERGE VERIFICATION ---")
    print(f"Dataset 1: {len(df_1)} participants")
    print(f"Dataset 2: {len(df_2)} participants")
    print(f"Merged dataset: {len(df_merged)} participants")
    print(f"Expected total: {len(df_1) + len(df_2)} participants")

    print(f"Unique IDs: {df_merged['ID'].nunique()}")
    print(f"Duplicate IDs: {df_merged['ID'].duplicated().sum()}")

    print("\nMerged participant IDs:")
    print(df_merged["ID"].tolist())

    # Save merged dataset
    df_merged.to_excel(output_path, index=False)

    print(f"\nMerged dataset saved to: {output_path}")

    return df_merged


if __name__ == "__main__":

    path_1 = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\final_dataframe_valids_age_groups.xlsx"
    path_2 = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\invalidos\final_dataframe_invalids_age_groups.xlsx"
    output_path = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\report_participants_all.xlsx"

    merge_participant_data(path_1, path_2, output_path)
