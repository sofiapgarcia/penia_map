"""
Functions to interpolate the data.

Available Functions
-------------------

[Public]
interpolate_signal(...): Fill missing samples and interpolate signal values.
fill_missing_samples(...): Detects sequence gaps and inserts missing samples
interpolate_missing_values(...): Interpolates sensor signals using spline and linear methods

[Private]
_fill_nan_linear(...): Applies linear interpolation to signal columns
_fill_acc_nan_spline(...): Applies cubic spline interpolation to accelerometer signals
_create_missing_rows(...): Creates placeholder rows for detected sequence gaps
_insert_missing_rows(...): Inserts generated missing rows into the DataFrame
_unfold_sequence_counter(...): Converts wrapped nSeq counter into a continuous sequence
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

# internal imports
from data_loader.data_loading import analyze_sequence

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def interpolate_signal(df):
    """
    Fill missing samples and interpolate signal values.

    Combines gap detection/filling (fill_missing_samples) with
    interpolation (interpolate_missing_values) into a single step.

    :param df: Input DataFrame containing the nSeq sequence counter and signal columns
    :return: Tuple containing the interpolated DataFrame, sequence analysis information,
             and a boolean mask DataFrame marking interpolated values
    """

    df_filled, info = _fill_missing_samples(df)

    interpolated_mask = df_filled.isna()

    df_interpolated = _interpolate_missing_values(df_filled)

    return df_interpolated, info, interpolated_mask


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _fill_missing_samples(df):
    """
    Fill missing samples in the signal sequence by detecting gaps in nSeq.

    :param df: Input DataFrame containing the nSeq sequence counter and signal columns
    :return: Tuple containing the corrected DataFrame and sequence analysis information
    """

    # Reset index to ensure correct positional indexing
    df = df.reset_index(drop=True)

    # Analyze sequence integrity
    analysis = analyze_sequence(df["nSeq"])

    # Create missing rows based on detected sequence gaps
    missing_rows = _create_missing_rows(
        df,
        analysis["invalid_jumps"]
    )

    # Insert missing samples into DataFrame, if any
    if missing_rows:
        df_filled = _insert_missing_rows(
            df,
            missing_rows
        )
    else:
        df_filled = df

    # Always unfold nSeq into a continuous sequence,
    # regardless of whether missing samples were found —
    # wraps happen independently of gaps.
    df_filled["nSeq"] = _unfold_sequence_counter(
        df_filled["nSeq"]
    )

    return df_filled, {
        "inserted_samples": len(missing_rows),
        "analysis": analysis
    }


def _interpolate_missing_values(df):
    """
    Interpolate missing values according to signal type.

    Accelerometer channels (x, target, z) are interpolated using cubic spline,
    while all remaining signal channels are interpolated using linear interpolation.

    :param df: Input DataFrame containing sensor signals
    :return: DataFrame with interpolated missing values
    """

    df_filled = df.copy()

    # Accelerometer channels
    ACC_COLUMNS = ["x", "y", "z", "gACCx", "gACCy", "gACCz"]

    acc_columns = [
        column
        for column in ACC_COLUMNS
        if column in df.columns
    ]

    # All remaining signal columns except sequence counter
    other_columns = [
        column
        for column in df.columns
        if column != "nSeq"
        and column not in acc_columns
    ]

    # Apply cubic spline to accelerometer
    if acc_columns:

        df_filled = _fill_nan_linear(df_filled, acc_columns)

    # Apply linear interpolation to all other signals
    if other_columns:
        df_filled = _fill_nan_linear(df_filled, other_columns)

    return df_filled

def _fill_nan_linear(df, columns):
    """
    Fill missing values in signal columns using linear interpolation.

    ```
    :param df: Input DataFrame containing signal columns
    :param columns: Columns to interpolate
    :return: DataFrame with interpolated values
    """

    df_filled = df.copy()


    for column in columns:

        df_filled[column] = (
            df_filled[column]
            .interpolate(
                method="linear"
            )
        )

    return df_filled

def _fill_acc_nan_spline(df, acc_columns):
    """
    Fill missing values in accelerometer signals using natural cubic spline interpolation.

    ```
    :param df: Input DataFrame containing accelerometer signals
    :param acc_columns: List of accelerometer column names
    :return: DataFrame with accelerometer NaN values filled
    """

    df_filled = df.copy()


    x = np.arange(len(df_filled))


    for column in acc_columns:

        y = df_filled[column].values
        valid = ~np.isnan(y)

        spline = CubicSpline(
            x[valid],
            y[valid],
            bc_type="natural"
        )

        df_filled[column] = spline(x)


    return df_filled

def _create_missing_rows(df, invalid_jumps):
    """
    Create empty rows for missing sequence values.

    :param df: Input DataFrame
    :param invalid_jumps: Invalid jumps detected in nSeq
    :return: List containing insertion positions and missing rows
    """

    missing_rows = []

    sequence = df["nSeq"]

    for idx, jump in invalid_jumps.items():

        previous = int(sequence.iloc[idx - 1])
        current = int(sequence.iloc[idx])

        # Normal sequence gap
        if jump > 1:
            missing_values = range(
                previous + 1,
                current
            )

        # Counter overflow case (32767 -> 0)
        else:
            missing_values = (
                list(range(previous + 1, 32768))
                +
                list(range(0, current))
            )


        for value in missing_values:

            row = {
                column: np.nan
                for column in df.columns
            }

            row["nSeq"] = value

            missing_rows.append(
                (idx, row)
            )

    return missing_rows



def _insert_missing_rows(df, missing_rows):
    """
    Insert missing rows into original DataFrame.

    :param df: Original DataFrame
    :param missing_rows: Missing rows with insertion positions
    :return: DataFrame containing inserted samples
    """

    pieces = []
    last_position = 0

    for position in sorted(
        set(pos for pos, _ in missing_rows)
    ):

        # Original samples before missing data
        pieces.append(
            df.iloc[last_position:position]
        )

        rows = [
            row
            for pos, row in missing_rows
            if pos == position
        ]

        # Missing samples
        pieces.append(
            pd.DataFrame(rows)
        )


        last_position = position


    # Remaining original samples
    pieces.append(
        df.iloc[last_position:]
    )

    return pd.concat(
        pieces,
        ignore_index=True
    )


def _unfold_sequence_counter(sequence):
    """
    Convert wrapped nSeq counter into continuous sequence.

    :param sequence: Sequence counter containing possible resets
    :return: Continuous sequence counter
    """

    offset = 0
    unfolded = []

    for i, value in enumerate(sequence):

        # Detect counter reset
        if (
            i > 0 and
            value < sequence.iloc[i-1]
        ):

            offset += 32768

        unfolded.append(
            value + offset
        )

    return unfolded