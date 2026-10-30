"""
Functions to build the final single-row feature summary of a subject from its
power, coordination, band power and IHD_values data.

Available Functions
-------------------
[Public]
build_subject_feature_row(subject, power_info, ihd_values, ...): Combine all feature sources into a single-row DataFrame for one subject.

[Private]
_get_rep_row(df, marker, rep_col): Extract the row of a given repetition marker (e.g. peak power or max force repetition).
_get_mean_row(df, rep_col): Compute the mean of every column across all repetitions.
_get_slope_row(df, value_col): Compute the slope of a column across repetitions using a linear fit.
_get_ihd_values_summary(ihd_values): Compute the max and mean of the IHD_values.
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def build_subject_feature_row(subject, power_info, ihd_values,
                              coord_dhd=None, coord_ihd=None,
                              band_power_dhd=None, band_power_ihd=None,
                              median_freq_cols=("Median_Frequency_extensor",
                                                "Median_Frequency_flexor")):
    """
    Build the final single-row feature summary for one subject by combining
    power_info with the peak-power repetition row (DHD) and max-force
    repetition row (IHD) from the coordination and band power metrics, the
    mean of each metric across all repetitions, the slope of the median
    frequency across repetitions, and the max/mean summary of IHD_values.

    :param subject: Subject identifier to attach to the resulting row
    :param power_info: Single-row DataFrame with peak power info, including
        the rep_at_peak_power column
    :param coord_dhd: Coordination metrics DataFrame (DHD), one row per repetition
    :param coord_ihd: Coordination metrics DataFrame (IHD), one row per repetition
    :param band_power_dhd: Band power info DataFrame (DHD), one row per repetition
    :param band_power_ihd: Band power info DataFrame (IHD), one row per repetition
    :param ihd_values: Series or single-column DataFrame with the subject's
        IHD_values (e.g. subject_data["IHD_values"]), used to compute the
        max and mean dynapenia features
    :param median_freq_cols: Names of the median-frequency columns inside
        band_power_dhd / band_power_ihd (one per muscle, extensor and flexor),
        each used to compute its own across-repetition slope
    :return: Single-row DataFrame with all combined features for the subject,
        prefixed by source (coord_dhd_, band_power_dhd_, coord_dhd_mean_, etc.)
    """

    # Drop rep_at_peak_power since it's just a marker, not a feature
    power_info_clean = power_info.drop(columns=["rep_at_peak_power"]).reset_index(drop=True)

    # DHD features come from the repetition marked as the peak power repetition
    coord_dhd_row      = _get_rep_row(coord_dhd, "rep_peakpower").add_prefix("rep_pp_dhd_")
    band_power_dhd_row = _get_rep_row(band_power_dhd, "rep_peakpower").add_prefix("rep_pp_dhd_")

    # IHD features come from the repetition marked as the max force repetition
    coord_ihd_row      = _get_rep_row(coord_ihd, "rep_max").add_prefix("rep_max_ihd_")
    band_power_ihd_row = _get_rep_row(band_power_ihd, "rep_max").add_prefix("rep_max_ihd_")

    # Mean of every metric across all repetitions (DHD and IHD domains)
    coord_dhd_mean_row      = _get_mean_row(coord_dhd).add_prefix("mean_dhd_")
    band_power_dhd_mean_row = _get_mean_row(band_power_dhd).add_prefix("mean_dhd_")
    coord_ihd_mean_row      = _get_mean_row(coord_ihd).add_prefix("mean_ihd_")
    band_power_ihd_mean_row = _get_mean_row(band_power_ihd).add_prefix("mean_ihd_")

    # Slope of the median frequency across repetitions (DHD and IHD domains),
    # computed separately for each muscle (extensor / flexor)
    band_power_dhd_slope_rows = [
        _get_slope_row(band_power_dhd, col).add_prefix("dhd_")
        for col in median_freq_cols
    ]
    band_power_ihd_slope_rows = [
        _get_slope_row(band_power_ihd, col).add_prefix("ihd_")
        for col in median_freq_cols
    ]

    # Max/mean summary of IHD_values -> also a dynapenia (IHD domain) feature
    ihd_values_row = _get_ihd_values_summary(ihd_values).add_prefix("ihd_values_")

    # Combine all pieces side by side into a single row
    subject_row = pd.concat(
        [power_info_clean,
         coord_dhd_row, band_power_dhd_row,
         coord_ihd_row, band_power_ihd_row,
         coord_dhd_mean_row, band_power_dhd_mean_row,
         coord_ihd_mean_row, band_power_ihd_mean_row,
         *band_power_dhd_slope_rows, *band_power_ihd_slope_rows,
         ihd_values_row],
        axis=1
    )
    subject_row.insert(0, "subject", subject)

    return subject_row

# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _get_rep_row(df, marker, rep_col="repetition"):
    """
    Extract the single row corresponding to a given repetition marker from a
    dataframe that contains multiple rows per repetition, and drop the
    repetition column from the result.

    :param df: Input DataFrame with one row per repetition (e.g. coord_dhd,
        scalogram_dhd, band_power_dhd, or their ihd counterparts)
    :param marker: Value to match in the repetition column (e.g. "rep_peakpower"
        or "rep_max")
    :param rep_col: Name of the column identifying the repetition
    :return: Single-row DataFrame without the repetition column, or an empty
        DataFrame if df is None or the marker is not found
    """
    # Nothing to extract if the dataframe wasn't produced for this subject
    if df is None or df.empty or rep_col not in df.columns:
        return pd.DataFrame()

    # Keep only the row matching the marker, then drop the repetition column
    row = df[df[rep_col] == marker].drop(columns=[rep_col])

    if row.empty:
        print(f"  [warning] marker '{marker}' not found (column '{rep_col}')")
        return pd.DataFrame()

    # Reset index so concatenation later aligns correctly
    return row.reset_index(drop=True)


def _get_mean_row(df, rep_col="repetition"):
    """
    Compute the mean of every column across all repetitions of a dataframe.

    :param df: Input DataFrame with one row per repetition (e.g. coord_dhd,
        band_power_dhd, or their ihd counterparts)
    :param rep_col: Name of the column identifying the repetition, dropped
        before averaging since it's not a numeric feature
    :return: Single-row DataFrame with the mean of each remaining column, or
        an empty DataFrame if df is None
    """
    if df is None or df.empty:
        return pd.DataFrame()

    numeric_df = df.drop(columns=[rep_col], errors="ignore")

    return numeric_df.mean(numeric_only=True).to_frame().T.reset_index(drop=True)


def _get_slope_row(df, value_col):
    """
    Fit a first-degree polynomial (straight line) of value_col against the
    row position (0, 1, 2, ...) and return its slope. Rows are already in
    repetition order, so the row position is used as the x-axis directly --
    this avoids relying on the repetition column's exact numbering, which
    also contains a non-numeric marker (e.g. "rep_peakpower"/"rep_max") for
    one row.

    :param df: Input DataFrame with one row per repetition, in repetition order
    :param value_col: Name of the column to regress against row position
    :return: Single-row, single-column DataFrame named "<value_col>_slope",
        or an empty DataFrame if df is None, the column is missing, or there
        are fewer than 2 valid points to fit a line
    """
    if df is None or value_col not in df.columns:
        return pd.DataFrame()

    valid = df[value_col].dropna()

    if len(valid) < 2:
        print(f"  [warning] not enough points to fit slope for '{value_col}'")
        return pd.DataFrame()

    x = np.arange(1,len(valid)+1)
    slope, _intercept = np.polyfit(x, valid.to_numpy(), deg=1)

    return pd.DataFrame({f"{value_col}_slope": [slope]})


def _get_ihd_values_summary(ihd_values):
    """
    Compute the maximum and mean of the subject's IHD_values dictionary.

    :param ihd_values: Dictionary with keys such as H1, H2, H3 and the
        corresponding force values.
    :return: Single-row DataFrame with columns "max" and "mean".
    """
    if ihd_values is None:
        return pd.DataFrame()

    values = list(ihd_values.values())

    return pd.DataFrame({
        "max": [max(values)],
        "mean": [sum(values) / len(values)],
    })
