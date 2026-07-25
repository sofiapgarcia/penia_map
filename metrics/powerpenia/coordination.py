import pandas as pd

from constants import FORCE_COLUMN

def get_onset_difference(onset_extensor_idx, onset_flexor_idx, fs):
    """
    Difference (ms) between the extensor and flexor onset.
    Positive -> extensor onset happened AFTER flexor onset.
    Negative -> extensor onset happened BEFORE flexor onset.

    :param onset_extensor_idx: Sample index of extensor onset
    :param onset_flexor_idx: Sample index of flexor onset
    :param fs: Sampling frequency (Hz)
    :return: Time difference in milliseconds
    """
    diff_samples = onset_extensor_idx - onset_flexor_idx
    return (diff_samples / fs) * 1000


def get_cocontraction_time(onset_extensor_idx, onset_flexor_idx, concentric_end_idx, fs):
    """
    Time (ms) between the onset of whichever muscle activated LAST and the
    end of the concentric phase.

    :param onset_extensor_idx: Sample index of extensor onset
    :param onset_flexor_idx: Sample index of flexor onset
    :param concentric_end_idx: Sample index where concentric phase ends
    :param fs: Sampling frequency (Hz)
    :return: Co-contraction duration in milliseconds
    """
    last_onset_idx = max(onset_extensor_idx, onset_flexor_idx)
    cocontraction_samples = concentric_end_idx - last_onset_idx
    return (cocontraction_samples / fs) * 1000


def get_emg_peak_in_phase(df_normalized, phase_start_idx, phase_end_idx, emg_column):
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


def get_envelope_peak_delay(peak_idx_extensor, peak_idx_flexor, fs):
    """
    Envelope Peak Delay (ms): temporal difference between the agonist (extensor)
    and antagonist (flexor) envelope peaks.
    Positive -> extensor envelope peak happened AFTER flexor envelope peak.
    Negative -> extensor envelope peak happened BEFORE flexor envelope peak.

    :param peak_idx_extensor: Sample index of the extensor envelope peak
    :param peak_idx_flexor: Sample index of the flexor envelope peak
    :param fs: Sampling frequency (Hz)
    :return: Envelope peak delay in milliseconds
    """
    diff_samples = peak_idx_extensor - peak_idx_flexor
    return (diff_samples / fs) * 1000
def get_peak_rate_of_force_development(
    df,
    start_idx,
    end_idx,
    force_column,
    fs,
):
    """
    Compute the Peak Rate of Force Development (Peak RFD) during a given phase.

    The Peak RFD is defined as the maximum first derivative of the force signal
    within the specified interval.

    :param df: DataFrame containing the force signal.
    :param start_idx: Start index of the phase.
    :param end_idx: End index of the phase.
    :param force_column: Name of the force column.
    :param fs: Sampling frequency (Hz).
    :return: Peak Rate of Force Development (N/s).
    """
    force = df.loc[start_idx:end_idx, force_column].to_numpy()

    derivative = np.gradient(force) * fs

    return float(np.max(derivative))

def get_muscle_coordination_info(
    subject_data_normalized,
    onset_df,
    power_info,
    subject_id,
    condition="HD",
    fs=None,
    onset_extensor_column="EMG_Extensor_filt",
    onset_flexor_column="EMG_Flexor_filt",
    emg_extensor_column="EMG_Extensor_norm",
    emg_flexor_column="EMG_Flexor_norm",
):
    """
    Build the full muscle coordination info for a given condition (only 'HD'
    supported for now) using the onset DataFrame from detect_emg_onsets.

    :param subject_data_normalized: Dict output of normalize_emg containing 'data' and 'params'
    :param onset_df: DataFrame output for a condition from detect_emg_onsets
    :param power_info: DataFrame containing a 'rep_at_peak_power' column
    :param subject_id: Subject identifier tag
    :param condition: Condition name to process (default 'HD')
    :param fs: Sampling frequency (Hz). If None, extracted from subject_data_normalized
    :param onset_extensor_column: Muscle column name used for extensor onsets
    :param onset_flexor_column: Muscle column name used for flexor onsets
    :param emg_extensor_column: Column name of normalized extensor EMG
    :param emg_flexor_column: Column name of normalized flexor EMG
    :return: Tuple (df_hd, df_hi) containing coordination DataFrames
    """
    if condition != "HD":
        raise NotImplementedError(
            "get_muscle_coordination_info currently only supports condition='HD'."
        )

    if fs is None:
        fs = subject_data_normalized[condition]["params"]["sampling rate"]

    df_normalized = subject_data_normalized[condition]["data"]
    rep_at_peak_power = int(power_info["rep_at_peak_power"].iloc[0])

    coordination_info = {}

    # Group rows by repetition number from the onset DataFrame
    repetition_groups = onset_df.groupby("rep")

    for rep, rep_data in repetition_groups:
        ext_row = rep_data[rep_data["muscle"] == onset_extensor_column]
        flex_row = rep_data[rep_data["muscle"] == onset_flexor_column]

        if ext_row.empty or flex_row.empty:
            continue

        onset_extensor_idx = ext_row["onset_idx"].values[0]
        onset_flexor_idx = flex_row["onset_idx"].values[0]

        # Extract concentric phase boundaries (shared between muscles in HD condition)
        concentric_start_idx = ext_row["concentric_start_idx"].values[0]
        concentric_end_idx = ext_row["concentric_end_idx"].values[0]
        concentric_duration_ms = (concentric_end_idx - concentric_start_idx) / fs * 1000

        onset_diff_ms = get_onset_difference(onset_extensor_idx, onset_flexor_idx, fs)
        onset_diff_ms_abs = abs(onset_diff_ms)

        cocontraction_ms = get_cocontraction_time(onset_extensor_idx, onset_flexor_idx, concentric_end_idx, fs)
        cocontraction_normalized = cocontraction_ms / concentric_duration_ms

        # EMG peak for both muscles
        peak_idx_extensor, emg_peak_extensor = get_emg_peak_in_phase(
            df_normalized, concentric_start_idx, concentric_end_idx, emg_extensor_column
        )
        peak_idx_flexor, emg_peak_flexor = get_emg_peak_in_phase(
            df_normalized, concentric_start_idx, concentric_end_idx, emg_flexor_column
        )
        emg_peak_ratio = emg_peak_extensor / emg_peak_flexor

        # Envelope Peak Delay metrics
        envelope_peak_delay_ms = get_envelope_peak_delay(peak_idx_extensor, peak_idx_flexor, fs)
        envelope_peak_delay_ms_abs = abs(envelope_peak_delay_ms)
        normalized_envelope_peak_delay = envelope_peak_delay_ms / concentric_duration_ms
        normalized_envelope_peak_delay_abs = envelope_peak_delay_ms_abs / concentric_duration_ms

        # Rate of Force Development Peak
        peak_rfd = get_peak_rate_of_force_development(
            df_normalized,
            concentric_start_idx,
            concentric_end_idx,
            FORCE_COLUMN,
            fs,
        )

        rep_key = "rep_peakpower" if rep == rep_at_peak_power else rep

        coordination_info[rep_key] = {
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
            "peak_rfd": peak_rfd,
        }

    df_hd = build_coordination_dataframe({condition: coordination_info}, subject_id)
    df_hi = None

    return df_hd, df_hi


def build_coordination_dataframe(coordination_info, subject_id=None, condition="HD"):
    """
    Flatten coordination_info into a tidy DataFrame with one row per repetition.

    :param coordination_info: Dict containing metric values per repetition key
    :param subject_id: Optional subject identifier to prepending as first column
    :param condition: Condition key inside coordination_info (default 'HD')
    :return: Tidy DataFrame with repetition metrics
    """
    rows = []
    for rep_key, metrics in coordination_info[condition].items():
        row = {"repetition": rep_key}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)

    if subject_id is not None:
        df.insert(0, "subject_id", subject_id)

    return df