"""
Functions to load and process OpenSignals data and analyze signal sequence integrity.

Available Functions
-------------------
[Public]
load_subject_data(...): Load signal and auxiliary data for a subject.
analyze_sequence(...): Analyze signal sequence integrity and data loss.

[Private]
extract_data_from_file(...): Extract metadata and signals from an OpenSignals file.
_parse_filename(...): Extract the experimental condition from a filename.
_load_handgrip_position(...): Load handgrip position data.
_load_handgrip_isometric(...): Load isometric handgrip measurements.
_get_levels(...): Estimate low and high signal levels.
_plot_levels(...): Plot the signal with estimated levels.
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import pandas as pd
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def load_subject_data(base_path, subject):
    """
        Load all OpenSignals files for a subject into a structured dictionary.
        Also loads auxiliary Excel data (potentiometer and handgrip data).

        :param base_path: Path to the folder containing all subjects' folders
        :param subject: Subject ID (e.g., "S83")
        :return: Dictionary keyed by condition
        """

    subject_path = os.path.join(base_path, subject)
    subject_id = subject.replace("S", "")

    hp_value, hp_source = _load_handgrip_position(base_path, subject_id)
    hg_values = _load_handgrip_isometric(base_path, subject_id)

    subject_data = {
        "handgrip_position": hp_value,
        "handgrip_position_source": hp_source,
        "HI_values": hg_values
    }

    for filename in os.listdir(subject_path):

        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(subject_path, filename)

        params, df = extract_data_from_file(file_path)

        condition = _parse_filename(filename)

        if condition == "calibration":
            # Only the two displacement levels are needed for calibration,
            # not the full raw signal
            low_level, high_level = _get_levels(df["DISPLACEMENT"], show_plot = False)

            subject_data[condition] = {
                "params": params,
                "low_level": low_level,
                "high_level": high_level
            }

        else:
            subject_data[condition] = {
                "params": params,
                "data": df
            }

    return subject_data

def analyze_sequence(sequence, fs=1000):
    """
    Analyze nSeq sequence integrity by detecting wraps, missing samples,
    invalid jumps, and computing signal loss and gap continuity metrics.

    :param sequence: pandas Series containing nSeq values from OpenSignals data
    :param fs: Sampling frequency in Hz
    :return: Dictionary with sequence integrity and gap continuity metrics
    """

    # Compute difference between consecutive samples
    diff = sequence.diff()

    # Detect valid counter wraps
    wraps = (diff == -32767).sum()

    # Identify gaps between consecutive samples
    gap_sizes = diff[diff > 1] - 1

    # Total number of missing samples
    missing_samples = gap_sizes.sum()

    # Number of distinct gaps
    number_of_gaps = len(gap_sizes)

    # Gap statistics
    if number_of_gaps > 0:
        max_consecutive_missing = gap_sizes.max()
        mean_gap_length = gap_sizes.mean()
        median_gap_length = gap_sizes.median()
        longest_gap_duration_s = max_consecutive_missing / fs
    else:
        max_consecutive_missing = 0
        mean_gap_length = 0
        median_gap_length = 0
        longest_gap_duration_s = 0

    # Estimate total expected number of samples
    expected_samples = wraps * 32768 + sequence.iloc[-1] + 1

    # Compute percentage of missing samples
    loss_percentage = (
        missing_samples / expected_samples * 100
        if expected_samples > 0 else 0
    )

    # Percentage of expected samples corresponding to the longest gap
    max_gap_percentage = (
        max_consecutive_missing / expected_samples * 100
        if expected_samples > 0 else 0
    )

    # Flag file as invalid if more than 10% of samples are missing
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

        # Gap continuity metrics
        "number_of_gaps": int(number_of_gaps),
        "max_consecutive_missing": int(max_consecutive_missing),
        "mean_gap_length": round(mean_gap_length, 2),
        "median_gap_length": round(median_gap_length, 2),
        "longest_gap_duration_s": round(longest_gap_duration_s, 3),
        "max_gap_percentage": round(max_gap_percentage, 2),

        # Invalid jumps
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
    condition = _parse_filename(filename)

    # Find EMG columns
    emg_cols = [col for col in df.columns if col.upper() == "EMG"]

    if condition == "HD_EMG":

        # Rename first two EMG channels
        if len(emg_cols) == 2:

            emg_indices = [
                i for i, col in enumerate(df.columns)
                if col.upper() == "EMG"
            ]

            df.columns.values[emg_indices[0]] = "EMG_Extensor"
            df.columns.values[emg_indices[1]] = "EMG_Flexor"

        # Rename accelerometer channels
        if len(df.columns) >= 7:
            df.columns.values[-3:] = ["x", "y", "z"]

    elif condition == "HD":

        # Rename signal types for HD acquisition
        df.columns = [
            column.replace("MICRO", "DISPLACEMENT")
            .replace("RAW", "FORCE")
            for column in df.columns
        ]

        # Drop magnetometer columns, not needed for this condition
        columns_to_drop = ["pMAGx", "pMAGy", "pMAGz"]

        df = df.drop(
            columns=[col for col in columns_to_drop if col in df.columns]
        )

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

    elif condition == "calibration":

        df.columns = [
            column.replace("MICRO", "DISPLACEMENT")
            .replace("RAW", "FORCE")
            for column in df.columns
        ]

        # Drop magnetometer columns, not needed for calibration
        columns_to_drop = ["pMAGx", "pMAGy", "pMAGz"]

        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    df = df.drop(columns="DI")

    params = {
        "sampling rate": device_info["sampling rate"],
        "date": device_info["date"],
        "time": device_info["time"]
    }

    return params, df

# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _parse_filename(file):
    """
    Parse subject ID and condition from filename.

    :param file: Filename in format 'Sx_HI.txt', 'Sx_HD.txt' or 'Sx_MVC_FLEXOR.txt'
    :return: condition (str): experimental test (HI, HD or MVC_FLEXOR)
    """
    name = file.replace(".txt", "")

    _ , condition = name.split("_", 1)

    return condition

def _load_handgrip_position(base_path, subject_id):
    """
    Load potentiometer position for a given subject.
    Tries 'pega_positions.xlsx' first; if the subject is not found there,
    falls back to 'potentiometer_positions.xlsx'.

    :param base_path: Path to dataset folder
    :param subject_id: Subject ID without "S" prefix
    :return: Tuple (position value or None, source label or None)
             source label is "handle" if found in pega_positions.xlsx,
             or "potentiometer" if found in potentiometer_positions.xlsx
    """

    # Try pega_positions.xlsx first
    pega_path = os.path.join(base_path, "pega_positions.xlsx")
    df_pega = pd.read_excel(pega_path)
    df_pega[df_pega.columns[0]] = df_pega[df_pega.columns[0]].astype(str)

    match = df_pega[df_pega.iloc[:, 0] == subject_id]

    if not match.empty:
        return match.iloc[0, 1], "handle"

    # Fallback to potentiometer_positions.xlsx
    pot_path = os.path.join(base_path, "potentiometer_positions.xlsx")
    df_pot = pd.read_excel(pot_path)
    df_pot[df_pot.columns[0]] = df_pot[df_pot.columns[0]].astype(str)

    match = df_pot[df_pot.iloc[:, 0] == subject_id]

    if not match.empty:
        return match.iloc[0, 1], "potentiometer"

    return None, None

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
    df[df.columns[0]] = df[df.columns[0]].astype(str)

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



def _get_levels(signal, pct=10 , show_plot = False):
    """
    Estimate the low and high levels of the signal.

    :param signal: 1D array (or list) containing the signal samples, e.g. in bits
    :param pct: Percentage of samples (from each end of the sorted signal) used to estimate each level
    :return: Tuple (low_level, high_level) with the estimated low and high levels
    """
    # Sort the signal to group low, transition, and high samples together
    s = np.sort(np.asarray(signal))
    n = len(s)

    # Number of samples used to estimate each level
    k = max(1, int(n * pct / 100))

    # Median of the lowest k samples -> low level estimate
    low_level = np.median(s[:k])

    # Median of the highest k samples -> high level estimate
    high_level = np.median(s[-k:])

    if show_plot:
        _plot_levels(signal, low_level, high_level)

    return low_level, high_level


def _plot_levels(signal, low_level, high_level):
    """
    Plot a signal together with its estimated low and high levels, for visual confirmation.

    :param signal: 1D array (or list) containing the signal samples, e.g. in bits
    :param low_level: Estimated low level, plotted as a dashed horizontal line
    :param high_level: Estimated high level, plotted as a dashed horizontal line
    :return: None
    """
    plt.figure(figsize=(12, 4))

    plt.plot(np.asarray(signal), label="Signal")

    plt.axhline(low_level, color="red", linestyle="--", label=f"Low level = {low_level:.1f}")
    plt.axhline(high_level, color="green", linestyle="--", label=f"High level = {high_level:.1f}")

    plt.xlabel("Sample")
    plt.ylabel("Displacement")
    plt.title("Calibration signal - estimated levels")
    plt.legend()
    plt.tight_layout()
    plt.show()