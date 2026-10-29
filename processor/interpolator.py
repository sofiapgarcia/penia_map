"""
Functions to interpolate the data.

Available Functions
-------------------
[Public]
interpolate_signal(...): Fill missing samples and interpolate signal values.

[Private]
_fill_missing_samples(...): Detects sequence gaps and inserts missing samples.
_interpolate_missing_values(...): Interpolates missing values in all signal columns.
_create_missing_rows(...): Creates placeholder rows for detected sequence gaps.
_insert_missing_rows(...): Inserts generated missing rows into the DataFrame.
_unfold_sequence_counter(...): Converts wrapped nSeq counter into a continuous sequence.
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import pandas as pd

# internal imports
from data_loader.data_loading import analyze_sequence

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def interpolate_signal(df):
    """
    Fill missing samples and interpolate signal values.
    Combines gap detection/filling with linear interpolation.
    :param df: Input DataFrame containing the nSeq sequence counter and signal columns
    :return: Tuple containing the interpolated DataFrame, sequence analysis information, and an interpolation mask
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
    Interpolate missing values in all signal columns using linear interpolation.
    :param df: Input DataFrame containing sensor signals
    :return: DataFrame with interpolated missing values
    """
    df_filled = df.copy()

    signal_columns = [
        column
        for column in df.columns
        if column != "nSeq"
    ]

    df_filled[signal_columns] = df_filled[signal_columns].interpolate(
        method="linear"
    )

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