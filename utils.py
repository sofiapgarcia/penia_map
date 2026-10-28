"""
Functions to build the final single-row feature summary of a subject from its
power, coordination, band power and HI_values data.

Available Functions
-------------------
[Public]
build_subject_feature_row(subject, power_info, hi_values, ...): Combine all feature sources into a single-row DataFrame for one subject.

[Private]
_get_rep_row(df, marker, rep_col): Extract the row of a given repetition marker (e.g. peak power or max force repetition).
_get_mean_row(df, rep_col): Compute the mean of every column across all repetitions.
_get_slope_row(df, value_col): Compute the slope of a column across repetitions using a linear fit.
_get_hi_values_summary(hi_values): Compute the max and mean of the HI_values.
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def build_subject_feature_row(subject, power_info, hi_values,
                              coord_hd=None, coord_hi=None,
                              band_power_hd=None, band_power_hi=None,
                              median_freq_cols=("Median_Frequency_extensor",
                                                "Median_Frequency_flexor")):
    """
    Build the final single-row feature summary for one subject by combining
    power_info with the peak-power repetition row (HD) and max-force
    repetition row (HI) from the coordination and band power metrics, the
    mean of each metric across all repetitions, the slope of the median
    frequency across repetitions, and the max/mean summary of HI_values.

    :param subject: Subject identifier to attach to the resulting row
    :param power_info: Single-row DataFrame with peak power info, including
        the rep_at_peak_power column
    :param coord_hd: Coordination metrics DataFrame (HD), one row per repetition
    :param coord_hi: Coordination metrics DataFrame (HI), one row per repetition
    :param band_power_hd: Band power info DataFrame (HD), one row per repetition
    :param band_power_hi: Band power info DataFrame (HI), one row per repetition
    :param hi_values: Series or single-column DataFrame with the subject's
        HI_values (e.g. subject_data["HI_values"]), used to compute the
        max and mean dynapenia features
    :param median_freq_cols: Names of the median-frequency columns inside
        band_power_hd / band_power_hi (one per muscle, extensor and flexor),
        each used to compute its own across-repetition slope
    :return: Single-row DataFrame with all combined features for the subject,
        prefixed by source (coord_hd_, band_power_hd_, coord_hd_mean_, etc.)
    """

    # Drop rep_at_peak_power since it's just a marker, not a feature
    power_info_clean = power_info.drop(columns=["rep_at_peak_power"]).reset_index(drop=True)

    # HD features come from the repetition marked as the peak power repetition
    coord_hd_row      = _get_rep_row(coord_hd, "rep_peakpower").add_prefix("rep_pp_hd_")
    band_power_hd_row = _get_rep_row(band_power_hd, "rep_peakpower").add_prefix("rep_pp_hd_")

    # HI features come from the repetition marked as the max force repetition
    coord_hi_row      = _get_rep_row(coord_hi, "rep_max").add_prefix("rep_max_hi_")
    band_power_hi_row = _get_rep_row(band_power_hi, "rep_max").add_prefix("rep_max_hi_")

    # Mean of every metric across all repetitions (HD and HI domains)
    coord_hd_mean_row      = _get_mean_row(coord_hd).add_prefix("mean_hd_")
    band_power_hd_mean_row = _get_mean_row(band_power_hd).add_prefix("mean_hd_")
    coord_hi_mean_row      = _get_mean_row(coord_hi).add_prefix("mean_hi_")
    band_power_hi_mean_row = _get_mean_row(band_power_hi).add_prefix("mean_hi_")

    # Slope of the median frequency across repetitions (HD and HI domains),
    # computed separately for each muscle (extensor / flexor)
    band_power_hd_slope_rows = [
        _get_slope_row(band_power_hd, col).add_prefix("hd_")
        for col in median_freq_cols
    ]
    band_power_hi_slope_rows = [
        _get_slope_row(band_power_hi, col).add_prefix("hi_")
        for col in median_freq_cols
    ]

    # Max/mean summary of HI_values -> also a dynapenia (HI domain) feature
    hi_values_row = _get_hi_values_summary(hi_values).add_prefix("hi_values_")

    # Combine all pieces side by side into a single row
    subject_row = pd.concat(
        [power_info_clean,
         coord_hd_row, band_power_hd_row,
         coord_hi_row, band_power_hi_row,
         coord_hd_mean_row, band_power_hd_mean_row,
         coord_hi_mean_row, band_power_hi_mean_row,
         *band_power_hd_slope_rows, *band_power_hi_slope_rows,
         hi_values_row],
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

    :param df: Input DataFrame with one row per repetition (e.g. coord_hd,
        scalogram_hd, band_power_hd, or their hi counterparts)
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

    :param df: Input DataFrame with one row per repetition (e.g. coord_hd,
        band_power_hd, or their hi counterparts)
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


def _get_hi_values_summary(hi_values):
    """
    Compute the maximum and mean of the subject's HI_values dictionary.

    :param hi_values: Dictionary with keys such as H1, H2, H3 and the
        corresponding force values.
    :return: Single-row DataFrame with columns "max" and "mean".
    """
    if hi_values is None:
        return pd.DataFrame()

    values = list(hi_values.values())

    return pd.DataFrame({
        "max": [max(values)],
        "mean": [sum(values) / len(values)],
    })
