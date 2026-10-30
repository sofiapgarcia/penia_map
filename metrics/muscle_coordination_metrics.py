"""
Functions for calculating muscle coordination metrics from EMG
signals during dynamic handgrip and isometric handgrip tasks.

Available Functions
-------------------
[Public]
get_muscle_coordination_metrics(...): Calculate muscle coordination metrics
    for the DHD and IHD conditions and return the results as separate DataFrames.

[Private]
_get_muscle_coordination_info_ihd(...): Calculate muscle coordination metrics
    for the IHD condition using the repetition with the maximum handgrip value.
_get_muscle_coordination_info_dhd(...): Calculate muscle coordination metrics
    for the DHD condition using the repetition at peak power.
_get_onset_difference(...): Calculate the time difference between extensor
    and flexor muscle activation onsets.
_get_cocontraction_time(...): Calculate co-contraction duration during the
    concentric phase of the DHD condition.
_get_emg_peak_in_phase(...): Identify the peak normalized EMG value and its
    sample index within a specified phase.
_get_envelope_peak_delay(...): Calculate the time difference between extensor
    and flexor EMG envelope peaks.
_get_peak_rate_of_force_development_window(...): Calculate the peak rate of
    force development over a sliding 20 ms window.
_get_emg_overlap_area(...): Calculate the overlap area between normalized
    extensor and flexor EMG envelopes.
_get_emg_rms(...): Calculate the root mean square of an EMG signal within
    a specified sample window.
_get_emg_integral(...): Calculate the integrated absolute EMG signal within
    a specified sample window.
_get_emg_mean(...): Calculate the mean absolute EMG signal within a
    specified sample window.
_get_cocontraction_overlap_ihd(...): Calculate the overlap duration between
    extensor and flexor activity during the IHD condition.
_get_activity_span_ihd(...): Calculate the total muscle activity duration
    during the IHD condition.
_get_max_repetition_ihd(...): Identify the IHD repetition with the maximum
    handgrip value.
_build_coordination_dataframe(...): Convert coordination metrics into a
    DataFrame with one row per repetition.
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import pandas as pd
import numpy as np

# internal imports
from constants import FORCE_COLUMN , FS

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def get_muscle_coordination_metrics(
    subject_data_normalized,
    onset_df_dhd,
    onset_df_ihd,
    power_info,
    onset_extensor_column="EMG_Extensor_filt",
    onset_flexor_column="EMG_Flexor_filt",
    emg_extensor_column="EMG_Extensor_norm",
    emg_flexor_column="EMG_Flexor_norm",
    offset_column="offset_idx",
    ihd_values_key="IHD_values",
):
    """
    Single entry point for main: runs the DHD and IHD coordination pipelines
    and returns both resulting DataFrames.

    :param subject_data_normalized: Dict output of normalize_emg containing 'data' and 'params' for both conditions
    :param onset_df_dhd: DataFrame output for the DHD condition from detect_emg_onsets
    :param onset_df_ihd: DataFrame output for the IHD condition from detect_emg_onsets
    :param power_info: DataFrame containing a 'rep_at_peak_power' column (used for the DHD condition)
    :param FS: Sampling frequency (Hz). If None, extracted from subject_data_normalized per condition
    :param onset_extensor_column: Muscle column name used for extensor onsets
    :param onset_flexor_column: Muscle column name used for flexor onsets
    :param emg_extensor_column: Column name of normalized extensor envelope EMG
    :param emg_flexor_column: Column name of normalized flexor envelope EMG
    :param offset_column: Column name of the muscle offset sample index (IHD onset_df)
    :param ihd_values_key: Key inside subject_data_normalized["IHD"] holding the H1/H2/H3 values
    :return: Tuple (df_dhd, df_ihd) with one row per repetition each
    """
    df_dhd = _get_muscle_coordination_info_dhd(subject_data_normalized, onset_df_dhd, power_info,
                                             onset_extensor_column=onset_extensor_column,
                                             onset_flexor_column=onset_flexor_column,
                                             emg_extensor_column=emg_extensor_column,
                                             emg_flexor_column=emg_flexor_column)

    df_ihd = _get_muscle_coordination_info_ihd(subject_data_normalized, onset_df_ihd,
                                             onset_extensor_column=onset_extensor_column,
                                             onset_flexor_column=onset_flexor_column,
                                             emg_extensor_column=emg_extensor_column,
                                             emg_flexor_column=emg_flexor_column, offset_column=offset_column,
                                             ihd_values_key=ihd_values_key)

    return df_dhd, df_ihd

# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #
def _get_muscle_coordination_info_ihd(
    subject_data_normalized,
    onset_df,
    onset_extensor_column="EMG_Extensor_filt",
    onset_flexor_column="EMG_Flexor_filt",
    emg_extensor_column="EMG_Extensor_norm",
    emg_flexor_column="EMG_Flexor_norm",
    offset_column="offset_idx",
    ihd_values_key="IHD_values",
):
    """
    Build the full muscle coordination info for the IHD (isometric) condition
    using the onset DataFrame from detect_emg_onsets.

    :param subject_data_normalized: Dict output of normalize_emg containing 'data' and 'params'
    :param onset_df: DataFrame output for the IHD condition from detect_emg_onsets
        (expected to contain, per rep/muscle row: "onset_idx" and ofFSet_column)
    :param onset_extensor_column: Muscle column name used for extensor onsets
    :param onset_flexor_column: Muscle column name used for flexor onsets
    :param emg_extensor_column: Column name of normalized extensor EMG
    :param emg_flexor_column: Column name of normalized flexor EMG
    :param offset_column: Column name of the muscle ofFSet sample index
    :param ihd_values_key: Key inside subject_data_normalized[condition] holding the H1/H2/H3 values
    :return: DataFrame with one row per repetition of IHD coordination metrics
    """

    df_normalized = subject_data_normalized["IHD"]["data"]
    rep_max = _get_max_repetition_ihd(subject_data_normalized, ihd_values_key=ihd_values_key)

    # keep only the repetition with the maximum IHD value
    rep_data = onset_df[onset_df["rep"] == rep_max]

    ext_row = rep_data[rep_data["muscle"] == onset_extensor_column]
    flex_row = rep_data[rep_data["muscle"] == onset_flexor_column]

    if ext_row.empty or flex_row.empty:
        raise ValueError(f"Missing extensor/flexor onset data for rep {rep_max}")

    onset_extensor_idx = ext_row["onset_idx"].values[0]
    onset_flexor_idx = flex_row["onset_idx"].values[0]
    offset_extensor_idx = ext_row[offset_column].values[0]
    offset_flexor_idx = flex_row[offset_column].values[0]

    onset_diff_ms = _get_onset_difference(onset_extensor_idx, onset_flexor_idx)
    onset_diff_ms_abs = abs(onset_diff_ms)

    cocontraction_ms = _get_cocontraction_overlap_ihd(onset_extensor_idx, offset_extensor_idx, onset_flexor_idx,
                                                     offset_flexor_idx)
    activity_span_ms = _get_activity_span_ihd(onset_extensor_idx, offset_extensor_idx, onset_flexor_idx,
                                             offset_flexor_idx)
    cocontraction_normalized = cocontraction_ms / activity_span_ms

    # EMG peak for both muscles, each within its own onset->offset window
    _, emg_peak_extensor = _get_emg_peak_in_phase(df_normalized, onset_extensor_idx, offset_extensor_idx,
                                                  emg_extensor_column)
    _, emg_peak_flexor = _get_emg_peak_in_phase(df_normalized, onset_flexor_idx, offset_flexor_idx, emg_flexor_column)
    emg_peak_ratio = emg_peak_extensor / emg_peak_flexor

    # RMS, integral (IEMG) and mean, each within its own onset->offset window
    rms_extensor = _get_emg_rms(df_normalized, onset_extensor_idx, offset_extensor_idx, emg_extensor_column)
    rms_flexor = _get_emg_rms(df_normalized, onset_flexor_idx, offset_flexor_idx, emg_flexor_column)

    iemg_extensor = _get_emg_integral(df_normalized, onset_extensor_idx, offset_extensor_idx, emg_extensor_column)
    iemg_flexor = _get_emg_integral(df_normalized, onset_flexor_idx, offset_flexor_idx, emg_flexor_column)

    mean_emg_extensor = _get_emg_mean(df_normalized, onset_extensor_idx, offset_extensor_idx, emg_extensor_column)
    mean_emg_flexor = _get_emg_mean(df_normalized, onset_flexor_idx, offset_flexor_idx, emg_flexor_column)

    # EMG overlap area (normalized signal): from the last onset to the
    # earliest offset among the two muscles (i.e. the actual window
    # where both muscles are simultaneously active). 0.0 if there is no
    # overlap at all.
    overlap_start_idx = max(onset_extensor_idx, onset_flexor_idx)
    overlap_end_idx = min(offset_extensor_idx, offset_flexor_idx)
    emg_overlap_area = _get_emg_overlap_area(df_normalized, overlap_start_idx, overlap_end_idx, emg_extensor_column,
                                             emg_flexor_column)
    emg_overlap_area_normalized = emg_overlap_area / activity_span_ms

    metrics = {
        "onset_diff_ms": onset_diff_ms,
        "onset_diff_ms_abs": onset_diff_ms_abs,
        "cocontraction_ms": cocontraction_ms,
        "cocontraction_normalized": cocontraction_normalized,
        "emg_peak_extensor": emg_peak_extensor,
        "emg_peak_flexor": emg_peak_flexor,
        "emg_peak_ratio": emg_peak_ratio,
        "rms_extensor": rms_extensor,
        "rms_flexor": rms_flexor,
        "iemg_extensor": iemg_extensor,
        "iemg_flexor": iemg_flexor,
        "mean_emg_extensor": mean_emg_extensor,
        "mean_emg_flexor": mean_emg_flexor,
        "emg_overlap_area": emg_overlap_area,
        "emg_overlap_area_normalized": emg_overlap_area_normalized,
    }

    df_ihd = _build_coordination_dataframe({"IHD": {"rep_max": metrics}}, condition="IHD")

    return df_ihd

def _get_muscle_coordination_info_dhd(
    subject_data_normalized,
    onset_df,
    power_info,
    onset_extensor_column="EMG_Extensor_filt",
    onset_flexor_column="EMG_Flexor_filt",
    emg_extensor_column="EMG_Extensor_norm",
    emg_flexor_column="EMG_Flexor_norm",
):
    """
    Build the full muscle coordination info for a given condition (DHD)
    using the onset DataFrame from detect_emg_onsets.

    :param subject_data_normalized: Dict output of normalize_emg containing 'data' and 'params'
    :param onset_df: DataFrame output for a condition from detect_emg_onsets
    :param power_info: DataFrame containing a 'rep_at_peak_power' column
    :param condition: Condition name to process (default 'DHD')
    :param onset_extensor_column: Muscle column name used for extensor onsets
    :param onset_flexor_column: Muscle column name used for flexor onsets
    :param emg_extensor_column: Column name of normalized extensor EMG
    :param emg_flexor_column: Column name of normalized flexor EMG
    :return: df_dhd containing coordination DataFrame
    """

    df_normalized = subject_data_normalized["DHD"]["data"]
    rep_at_peak_power = int(power_info["rep_at_peak_power"].iloc[0])

    rep_data = onset_df[onset_df["rep"] == rep_at_peak_power]
    ext_row = rep_data[rep_data["muscle"] == onset_extensor_column]
    flex_row = rep_data[rep_data["muscle"] == onset_flexor_column]

    if ext_row.empty or flex_row.empty:
        raise ValueError(f"Missing extensor/flexor onset data for rep {rep_at_peak_power}")

    onset_extensor_idx = ext_row["onset_idx"].values[0]
    onset_flexor_idx = flex_row["onset_idx"].values[0]

    # Extract concentric phase boundaries (shared between muscles in DHD condition)
    concentric_start_idx = ext_row["concentric_start_idx"].values[0]
    concentric_end_idx = ext_row["concentric_end_idx"].values[0]
    concentric_duration_ms = (concentric_end_idx - concentric_start_idx) / FS * 1000

    onset_diff_ms = _get_onset_difference(onset_extensor_idx, onset_flexor_idx)
    onset_diff_ms_abs = abs(onset_diff_ms)

    last_onset_idx = max(onset_extensor_idx, onset_flexor_idx)

    cocontraction_ms = _get_cocontraction_time(last_onset_idx, concentric_end_idx)
    cocontraction_normalized = cocontraction_ms / concentric_duration_ms

    # EMG peak for both muscles
    peak_idx_extensor, emg_peak_extensor = _get_emg_peak_in_phase(df_normalized, concentric_start_idx,
                                                                  concentric_end_idx, emg_extensor_column)
    peak_idx_flexor, emg_peak_flexor = _get_emg_peak_in_phase(df_normalized, concentric_start_idx, concentric_end_idx,
                                                              emg_flexor_column)
    emg_peak_ratio = emg_peak_extensor / emg_peak_flexor

    # Envelope Peak Delay metrics
    envelope_peak_delay_ms = _get_envelope_peak_delay(peak_idx_extensor, peak_idx_flexor)
    envelope_peak_delay_ms_abs = abs(envelope_peak_delay_ms)
    normalized_envelope_peak_delay = envelope_peak_delay_ms / concentric_duration_ms
    normalized_envelope_peak_delay_abs = envelope_peak_delay_ms_abs / concentric_duration_ms

    # Rate of Force Development Peak
    peak_rfd_window = _get_peak_rate_of_force_development_window(df_normalized, concentric_start_idx,
                                                                 concentric_end_idx, FORCE_COLUMN)

    # EMG overlap area (normalized signal): from the last onset to the
    # end of the concentric phase
    emg_overlap_area = _get_emg_overlap_area(df_normalized, last_onset_idx, concentric_end_idx, emg_extensor_column,
                                             emg_flexor_column)
    emg_overlap_area_normalized = emg_overlap_area / concentric_duration_ms

    metrics = {
        "onset_diff_ms": onset_diff_ms,
        "onset_diff_ms_abs": onset_diff_ms_abs,
        "cocontraction_ms": cocontraction_ms,
        "cocontraction_normalized": cocontraction_normalized,
        "emg_peak_extensor": emg_peak_extensor,
        "emg_peak_flexor": emg_peak_flexor,
        "emg_peak_ratio": emg_peak_ratio,
        "envelope_peak_delay_ms": envelope_peak_delay_ms,
        "envelope_peak_delay_ms_abs": envelope_peak_delay_ms_abs,
        "normalized_envelope_peak_delay": normalized_envelope_peak_delay,
        "normalized_envelope_peak_delay_abs": normalized_envelope_peak_delay_abs,
        "peak_rfd": peak_rfd_window,
        "emg_overlap_area": emg_overlap_area,
        "emg_overlap_area_normalized": emg_overlap_area_normalized,
    }

    df_dhd = _build_coordination_dataframe({"DHD": {"rep_peakpower": metrics}}, condition="DHD")

    return df_dhd


def _get_onset_difference(onset_extensor_idx, onset_flexor_idx):
    """
    Difference (ms) between the extensor and flexor onset.
    Positive -> extensor onset happened AFTER flexor onset.
    Negative -> extensor onset happened BEFORE flexor onset.

    :param onset_extensor_idx: Sample index of extensor onset
    :param onset_flexor_idx: Sample index of flexor onset
    :return: Time difference in milliseconds
    """
    diff_samples = onset_extensor_idx - onset_flexor_idx
    return (diff_samples / FS) * 1000


def _get_cocontraction_time(last_onset_idx, concentric_end_idx):
    """
    Time (ms) between the onset of whichever muscle activated LAST and the
    end of the concentric phase.

    :param onset_extensor_idx: Sample index of extensor onset
    :param onset_flexor_idx: Sample index of flexor onset
    :param concentric_end_idx: Sample index where concentric phase ends
    :param FS: Sampling frequency (Hz)
    :return: Co-contraction duration in milliseconds
    """
    cocontraction_samples = concentric_end_idx - last_onset_idx
    return (cocontraction_samples / FS) * 1000


def _get_emg_peak_in_phase(df_normalized, phase_start_idx, phase_end_idx, emg_column):
    """
    Peak (max) value AND its global sample index for one muscle's normalized
    EMG envelope, within a given sample window (by default the concentric
    phase of the repetition).

    :param df_normalized: DataFrame containing normalized EMG signals
    :param phase_start_idx: Start sample index of the phase window
    :param phase_end_idx: End sample index of the phase window
    :param emg_column: Column name of the normalized EMG signal
    :return: Tuple (peak_idx, peak_value) -- peak_idx is the global index (not
        positional) of the peak sample, peak_value is the max amplitude
    """
    phase_signal = df_normalized[emg_column].iloc[phase_start_idx : phase_end_idx + 1]
    peak_idx = phase_signal.idxmax()
    peak_value = phase_signal.max()
    return peak_idx, peak_value


def _get_envelope_peak_delay(peak_idx_extensor, peak_idx_flexor):
    """
    Envelope Peak Delay (ms): temporal difference between the agonist (extensor)
    and antagonist (flexor) envelope peaks.
    Positive -> extensor envelope peak happened AFTER flexor envelope peak.
    Negative -> extensor envelope peak happened BEFORE flexor envelope peak.

    :param peak_idx_extensor: Sample index of the extensor envelope peak
    :param peak_idx_flexor: Sample index of the flexor envelope peak
    :return: Envelope peak delay in milliseconds
    """
    diff_samples = peak_idx_extensor - peak_idx_flexor
    return (diff_samples / FS) * 1000


def _get_peak_rate_of_force_development_window(
    df,
    start_idx,
    end_idx,
    force_column,
):
    """
    Compute the Peak Rate of Force Development (Peak RFD) during a given phase.

    The Peak RFD is defined as the maximum average rate of force increase
    calculated over a sliding 20 ms window within the specified interval.

    :param df: DataFrame containing the force signal.
    :param start_idx: Start index of the phase.
    :param end_idx: End index of the phase.
    :param force_column: Name of the force column.
    :return: Peak Rate of Force Development (N/s).
    """
    force = df.loc[start_idx:end_idx, force_column].to_numpy()

    # Convert the 20 ms window into a number of samples
    window_samples = int(round(0.020 * FS))

    if len(force) <= window_samples:
        raise ValueError(
            "The selected phase is shorter than the 20 ms RFD window."
        )

    # Compute the average rate of force development over a sliding 20 ms window
    derivative = (
        force[window_samples:] - force[:-window_samples]
    ) / (window_samples / FS)

    peak_rfd = np.max(derivative)

    return float(peak_rfd)


def _get_emg_overlap_area(df_normalized, start_idx, end_idx, emg_extensor_column, emg_flexor_column):
    """
    Overlap (common) area between the extensor and flexor normalized EMG
    envelopes within a given sample window: sum, sample by sample, of the
    smaller of the two normalized signals at each instant
    (i.e. sum of min(EMG_extensor_norm, EMG_flexor_norm)).

    This quantifies how much both muscles were simultaneously active and at
    which relative magnitude -- a common "EMG overlap area" / co-contraction
    common-area index. Must always be computed on the normalized EMG signal.

    Callers typically also report a normalized version of this value,
    dividing it by the relevant phase duration (concentric phase duration
    for DHD, full activity span for IHD) to make it comparable across
    repetitions/subjects with different phase durations.

    If end_idx <= start_idx (no valid window / no overlap), returns 0.0.

    :param df_normalized: DataFrame containing normalized EMG signals
    :param start_idx: Start sample index of the window
    :param end_idx: End sample index of the window
    :param emg_extensor_column: Column name of normalized extensor EMG
    :param emg_flexor_column: Column name of normalized flexor EMG
    :return: Overlap area value (sum of the point-wise minimum) in the window
    """
    if end_idx <= start_idx:
        return 0.0

    ext_signal = df_normalized[emg_extensor_column].iloc[start_idx : end_idx + 1].to_numpy()
    flex_signal = df_normalized[emg_flexor_column].iloc[start_idx : end_idx + 1].to_numpy()
    overlap_signal = np.minimum(ext_signal, flex_signal)
    return float(np.sum(overlap_signal))

def _get_emg_rms(df_normalized, start_idx, end_idx, emg_column):
    """
    Root Mean Square (RMS) of a muscle's normalized EMG envelope within a
    given sample window.

    :param df_normalized: DataFrame containing normalized EMG signals
    :param start_idx: Start sample index of the window
    :param end_idx: End sample index of the window
    :param emg_column: Column name of the normalized EMG signal
    :return: RMS value of the envelope in the window
    """
    signal = df_normalized[emg_column].iloc[start_idx : end_idx + 1].to_numpy()
    return float(np.sqrt(np.mean(np.square(signal))))


def _get_emg_integral(df_normalized, start_idx, end_idx, emg_column):
    """
    Integrated EMG (IEMG): sum of the absolute values of a muscle's
    normalized EMG envelope within a given sample window.

    :param df_normalized: DataFrame containing normalized EMG signals
    :param start_idx: Start sample index of the window
    :param end_idx: End sample index of the window
    :param emg_column: Column name of the normalized EMG signal
    :return: Integrated EMG value (sum of moduli) in the window
    """
    signal = df_normalized[emg_column].iloc[start_idx : end_idx + 1].to_numpy()
    return float(np.sum(np.abs(signal)))


def _get_emg_mean(df_normalized, start_idx, end_idx, emg_column):
    """
    Mean value of a muscle's normalized EMG envelope within a given sample
    window.

    :param df_normalized: DataFrame containing normalized EMG signals
    :param start_idx: Start sample index of the window
    :param end_idx: End sample index of the window
    :param emg_column: Column name of the normalized EMG signal
    :return: Mean envelope value in the window
    """
    signal = df_normalized[emg_column].iloc[start_idx : end_idx + 1].to_numpy()
    return float(np.mean(np.abs(signal)))


def _get_cocontraction_overlap_ihd(onset_extensor_idx, offset_extensor_idx, onset_flexor_idx, offset_flexor_idx):
    """
    Co-contraction time (ms) for the IHD (isometric) condition: the duration
    during which BOTH muscles were simultaneously active, i.e. the overlap
    between the extensor's and the flexor's own [onset, ofFSet] windows.

    :param onset_extensor_idx: Sample index of extensor onset
    :param offset_extensor_idx: Sample index of extensor ofFSet
    :param onset_flexor_idx: Sample index of flexor onset
    :param offset_flexor_idx: Sample index of flexor ofFSet
    :param FS: Sampling frequency (Hz)
    :return: Overlap (co-contraction) duration in milliseconds
    """
    overlap_start_idx = max(onset_extensor_idx, onset_flexor_idx)
    overlap_end_idx = min(offset_extensor_idx, offset_flexor_idx)
    overlap_samples = max(overlap_end_idx - overlap_start_idx, 0)
    return (overlap_samples / FS) * 1000


def _get_activity_span_ihd(onset_extensor_idx, offset_extensor_idx, onset_flexor_idx, offset_flexor_idx):
    """
    Total activity span (ms) for the IHD (isometric) condition: from the
    onset of whichever muscle activated FIRST to the ofFSet of whichever
    muscle deactivated LAST. Used to normalize the co-contraction time.

    :param onset_extensor_idx: Sample index of extensor onset
    :param offset_extensor_idx: Sample index of extensor ofFSet
    :param onset_flexor_idx: Sample index of flexor onset
    :param offset_flexor_idx: Sample index of flexor ofFSet
    :return: Total activity span in milliseconds
    """
    span_start_idx = min(onset_extensor_idx, onset_flexor_idx)
    span_end_idx = max(offset_extensor_idx, offset_flexor_idx)
    span_samples = span_end_idx - span_start_idx
    return (span_samples / FS) * 1000


def _get_max_repetition_ihd(subject_data_normalized, ihd_values_key="IHD_values"):
    """
    Identify which repetition corresponds to the maximum value in the IHD
    condition.

    :param subject_data_normalized: Dict containing the IHD values.
    :param ihd_values_key: Key containing the H1/H2/H3 values.
    :return: Repetition number (int) corresponding to the max IHD value.
    """
    ihd_values = subject_data_normalized[ihd_values_key]

    values = [ihd_values["H1"], ihd_values["H2"], ihd_values["H3"]]

    return values.index(max(values)) + 1


def _build_coordination_dataframe(coordination_info, condition="DHD"):
    """
    Flatten coordination_info into a tidy DataFrame with one row per repetition.

    :param coordination_info: Dict containing metric values per repetition key
    :param subject_id: Optional subject identifier to prepending as first column
    :param condition: Condition key inside coordination_info (default 'DHD')
    :return: Tidy DataFrame with repetition metrics
    """
    rows = []
    for rep_key, metrics in coordination_info[condition].items():
        row = {"repetition": rep_key}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)

    return df