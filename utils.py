import pandas as pd


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
    if df is None:
        return pd.DataFrame()

    # Keep only the row matching the marker, then drop the repetition column
    row = df[df[rep_col] == marker].drop(columns=[rep_col])

    if row.empty:
        print(f"  [warning] marker '{marker}' not found (column '{rep_col}')")
        return pd.DataFrame()

    # Reset index so concatenation later aligns correctly
    return row.reset_index(drop=True)


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


def build_subject_feature_row(subject, power_info, coord_hd, coord_hi,
                               scalogram_hd, scalogram_hi,
                               band_power_hd, band_power_hi, hi_values):
    """
    Build the final single-row feature summary for one subject by combining
    power_info with the peak-power repetition row (HD) and max-force
    repetition row (HI) from the coordination, scalogram, and band power
    metrics, as well as the max/mean summary of HI_values.

    :param subject: Subject identifier to attach to the resulting row
    :param power_info: Single-row DataFrame with peak power info, including
        the rep_at_peak_power column
    :param coord_hd: Coordination metrics DataFrame (HD), one row per repetition
    :param coord_hi: Coordination metrics DataFrame (HI), one row per repetition
    :param scalogram_hd: Scalogram info DataFrame (HD), one row per repetition
    :param scalogram_hi: Scalogram info DataFrame (HI), one row per repetition
    :param band_power_hd: Band power info DataFrame (HD), one row per repetition
    :param band_power_hi: Band power info DataFrame (HI), one row per repetition
    :param hi_values: Series or single-column DataFrame with the subject's
        HI_values (e.g. subject_data["HI_values"]), used to compute the
        max and mean dynapenia features
    :return: Single-row DataFrame with all combined features for the subject,
        prefixed by source (coord_hd_, scalogram_hd_, band_power_hd_, etc.)
    """

    # Drop rep_at_peak_power since it's just a marker, not a feature
    power_info_clean = power_info.drop(columns=["rep_at_peak_power"]).reset_index(drop=True)

    # HD features come from the repetition marked as the peak power repetition
    coord_hd_row      = _get_rep_row(coord_hd, "rep_peakpower").add_prefix("coord_hd_")
    scalogram_hd_row  = _get_rep_row(scalogram_hd, "rep_peakpower").add_prefix("scalogram_hd_")
    band_power_hd_row = _get_rep_row(band_power_hd, "rep_peakpower").add_prefix("band_power_hd_")

    # HI features come from the repetition marked as the max force repetition
    coord_hi_row      = _get_rep_row(coord_hi, "rep_max").add_prefix("coord_hi_")
    scalogram_hi_row  = _get_rep_row(scalogram_hi, "rep_max").add_prefix("scalogram_hi_")
    band_power_hi_row = _get_rep_row(band_power_hi, "rep_max").add_prefix("band_power_hi_")

    # Max/mean summary of HI_values -> also a dynapenia (HI domain) feature
    hi_values_row = _get_hi_values_summary(hi_values).add_prefix("hi_values_")

    # Combine all pieces side by side into a single row
    subject_row = pd.concat(
        [power_info_clean, coord_hd_row, scalogram_hd_row, band_power_hd_row,
         coord_hi_row, scalogram_hi_row, band_power_hi_row, hi_values_row],
        axis=1
    )
    subject_row.insert(0, "subject", subject)

    return subject_row