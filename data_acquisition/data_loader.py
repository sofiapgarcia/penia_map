"""
Functions to load the data.

Available Functions
-------------------

[Public]
load_subject_data(...): Load all OpenSignals files for a subject into a structured dictionary.
analyze_sequence(...): Analyzes signal sequence integrity.

[Private]
_extract_data_from_file(...): Loads data from OpenSignals files.
_parse_filename(...): Parses information from filenames.

"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import pandas as pd
import os
import json

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def load_subject_data(base_path, subject):
    """
        Load all OpenSignals files for a subject into a structured dictionary.
        Also loads auxiliary Excel data (potentiometer and handgrip data).

        :param base_path: Path to the folder containing all subjects' folders
        :param subject: Subject ID (e.g., "S83")
        :return: Dictionary keyed by condition, e.g.:
                 {
                     "HI": {
                         "subject": ...,
                         "params": {...},
                         "data": df,
                         "potentiometer_position": ...,
                         "handgrip_isometric": {...}
                     },
                     ...
                 }
        """

    subject_path = os.path.join(base_path, subject)
    subject_id = subject.replace("S", "")

    pot_value = _load_potentiometer_position(base_path, subject_id)
    hg_values = _load_handgrip_isometric(base_path, subject_id)

    subject_data = {}

    for filename in os.listdir(subject_path):

        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(subject_path, filename)

        _ , params, df = extract_data_from_file(file_path)

        condition = filename.replace(".txt", "").split("_", 1)[1]

        subject_data[condition] = {
            "params": params,
            "data": df,
            "potentiometer_position": pot_value,
            "handgrip_isometric": hg_values
        }

    return subject_data

def analyze_sequence(sequence):
    """
    Analyze nSeq sequence integrity by detecting wraps, missing samples,
    invalid jumps, and computing signal loss percentage.

    :param sequence: pandas Series containing nSeq values from OpenSignals data
    :return: Dictionary with:
             - wraps: number of detected counter wraps
             - missing_samples: estimated number of missing samples due to gaps
             - expected_samples: total expected samples in the sequence
             - loss_percentage: percentage of missing samples
             - invalid_jumps: Series with abnormal jumps in nSeq
    """

    # Compute difference between consecutive samples
    diff = sequence.diff()

    # Detect valid counter wraps
    wraps = (diff == -32767).sum()

    # Estimate missing samples from sequence gaps
    missing_samples = (diff[diff > 1] - 1).sum()

    # Estimate total expected number of samples
    expected_samples = wraps * 32768 + sequence.iloc[-1] + 1

    # Compute percentage of missing samples
    loss_percentage = (
        missing_samples / expected_samples * 100
        if expected_samples > 0 else 0
    )

    invalid_file = loss_percentage > 10

    # Identify invalid jumps
    invalid_jumps = diff[
        (diff != 1) &
        (diff != -32767) &
        (~diff.isna())
    ]

    return {
        "wraps": int(wraps),
        "missing_samples": int(missing_samples),
        "expected_samples": int(expected_samples),
        "acquired_samples": int(len(sequence)),
        "loss_percentage": round(loss_percentage, 2),
        "invalid_jumps": invalid_jumps,
        "invalid_file": invalid_file
    }

def extract_data_from_file(file_path):
    """
    Extract metadata and signal data from an OpenSignals text file.

    :param file_path: Path to the OpenSignals .txt file
    :return:
        - subject (str): subject ID
        - params (dict): acquisition metadata
        - df (DataFrame): signal data
    """

    metadata = None
    data_start = 0

    with open(file_path, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):

        if line.startswith("# {"):
            metadata = json.loads(line[2:])

        if "EndOfHeader" in line:
            data_start = i + 1
            break

    device_info = list(metadata.values())[0]

    df = pd.read_csv(
        file_path,
        sep="\t",
        skiprows=data_start,
        header=None,
        usecols=range(len(device_info["column"]))
    )

    df.columns = device_info["column"]

    filename = os.path.basename(file_path)

    # Extract subject and condition from filename
    subject, condition = _parse_filename(filename)

    # Find EMG columns
    emg_cols = [col for col in df.columns if col.upper() == "EMG"]

    if condition == "HD_EMG":

        # Rename first two EMG channels
        if len(emg_cols) >= 2:

            emg_indices = [
                i for i, col in enumerate(df.columns)
                if col.upper() == "EMG"
            ]

            df.columns.values[emg_indices[0]] = "EMG_Extensor"
            df.columns.values[emg_indices[1]] = "EMG_Flexor"

        # Rename accelerometer channels
        if len(df.columns) >= 3:
            df.columns.values[-3:] = ["x", "y", "z"]

    elif condition == "HD":

        # Rename signal types for HD acquisition
        df.columns = [
            column.replace("MICRO", "DISPLACEMENT")
            .replace("RAW", "FORCE")
            for column in df.columns
        ]

    elif condition in ["HI", "MVC"]:

        # Find EMG columns
        emg_indices = [

            i for i, col in enumerate(df.columns)

            if col.upper() == "EMG"

        ]

        # Keep everything up to the second EMG (inclusive)
        if len(emg_indices) >= 2:
            last_emg_idx = emg_indices[1]

            df = df.iloc[:, :last_emg_idx + 1].copy()

            # Rename EMG columns
            df.columns.values[emg_indices[0]] = "EMG_Extensor"

            df.columns.values[emg_indices[1]] = "EMG_Flexor"

    params = {
        "sampling rate": device_info["sampling rate"],
        "date": device_info["date"],
        "time": device_info["time"]
    }

    return subject, params, df

# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _parse_filename(file):
    """
    Parse subject ID and condition from filename.

    :param file: Filename in format 'Sx_HI.txt', 'Sx_HD.txt' or 'Sx_MVC_FLEXOR.txt'
    :return: Tuple containing:
             - subject (str): subject ID (e.g., S1)
             - condition (str): experimental test (HI, HD or MVC_FLEXOR)
    """
    name = file.replace(".txt", "")

    subject_raw, condition = name.split("_", 1)

    subject = subject_raw.replace("S", "")

    return subject, condition

def _load_potentiometer_position(base_path, subject_id):
    """
    Load potentiometer position for a given subject.

    ```
    :param base_path: Path to dataset folder
    :param subject_id: Subject ID without "S" prefix
    :return: Potentiometer position value or None if not found
    ```
    """

    file_path = os.path.join(base_path, "potentiometer_positions.xlsx")
    df = pd.read_excel(file_path)

    # Ensure consistent ID format
    df.iloc[:, 0] = df.iloc[:, 0].astype(str)

    # Find matching subject row
    match = df[df.iloc[:, 0] == subject_id]

    if match.empty:
        return None

    return match.iloc[0, 1]

def _load_handgrip_isometric(base_path, subject_id):
    """
    Load handgrip isometric values (H1, H2, H3) for a given subject.

    :param base_path: Path to dataset folder
    :param subject_id: Subject ID without "S" prefix
    :return: Dictionary with H1, H2, H3 values or None if not found
    ```
    """

    file_path = os.path.join(base_path, "handgrip_isometric.xlsx")
    df = pd.read_excel(file_path)

    # Ensure consistent ID format
    df.iloc[:, 0] = df.iloc[:, 0].astype(str)

    # Find matching subject row
    match = df[df.iloc[:, 0] == subject_id]

    if match.empty:
        return None

    row = match.iloc[0]

    return {
        "H1": row.get("H1"),
        "H2": row.get("H2"),
        "H3": row.get("H3")
    }

