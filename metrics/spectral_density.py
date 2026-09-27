import numpy as np
import pandas as pd
from scipy.signal import welch
from constants import FS
from metrics.powerpenia.coordination import get_max_repetition_hi
import matplotlib.pyplot as plt

# np.trapezoid replaced the deprecated np.trapz in newer NumPy releases; fall
# back to np.trapz on older versions that don't have it yet.
_integrate = getattr(np, "trapezoid", None) or np.trapz


def compute_band_powers(signal, fs, band_edges=None, nperseg=None, plot=False):
    """
    Compute the power spectral density (Welch's method) of a 1D signal and
    integrate it into a set of frequency bands.

    :param signal: 1D array-like signal
    :param fs: Sampling frequency (Hz)
    :param band_edges: Sequence of band edges (Hz), e.g. [0, 50, 100, ..., 500]
        gives the bands [0-50), [50-100), ..., [450-500). Defaults to 50 Hz
        wide bands spanning 0-500 Hz (10 bands), which covers essentially all
        EMG spectral content.
    :param nperseg: Segment length passed to scipy.signal.welch. Defaults to
        min(256, len(signal)) since analysis windows here are short (onset to
        concentric-phase-end / onset to offset of a single repetition).
    :return: Tuple (band_powers, freqs, psd) -- band_powers is a dict keyed by
        a "<low>-<high>Hz" label with the integrated power (trapezoidal rule)
        in that band; freqs/psd are the raw Welch output, in case the caller
        wants the full spectrum.
    """
    signal = np.asarray(signal, dtype=float)

    if band_edges is None:
        band_edges = np.arange(20, 500, 50)

    if nperseg is None:
        nperseg = len(signal)//3

    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)

    band_powers = {}
    for low, high in zip(band_edges[:-1], band_edges[1:]):
        mask = (freqs >= low) & (freqs < high)
        band_label = f"{low}-{high}Hz"
        band_powers[band_label] = float(_integrate(psd[mask], freqs[mask])) if mask.any() else 0.0

    if plot:
        plt.figure(figsize=(8, 4))
        plt.semilogy(freqs, psd)
        for edge in band_edges:
            plt.axvline(edge, color="gray", linestyle="--", linewidth=0.5)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Spectral Density")
        plt.title("Power Spectrum")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return band_powers, freqs, psd


def compute_median_frequency(freqs, psd):
    """
    Compute the median frequency (Median_Frequency) of a power spectrum: the frequency
    that splits the total spectral power into two equal halves (cumulative
    sum of the PSD reaches 50% of the total).

    :param freqs: 1D array of frequency bins (as returned by scipy.signal.welch)
    :param psd: 1D array of power spectral density values, same length as freqs
    :return: Median frequency (Hz) as a float. Returns 0.0 if the spectrum has
        no power (e.g. an all-zero signal).
    """
    freqs = np.asarray(freqs)
    psd = np.asarray(psd)

    total_power = np.sum(psd)
    if total_power <= 0:
        return 0.0

    cumulative_power = np.cumsum(psd)
    median_idx = np.searchsorted(cumulative_power, total_power / 2.0)
    median_idx = min(median_idx, len(freqs) - 1)

    return float(freqs[median_idx])


def compute_peak_frequency(freqs, psd):
    """
    Compute the peak frequency (Peak_Frequency) of a power spectrum: the frequency at
    which the PSD is maximal.

    :param freqs: 1D array of frequency bins (as returned by scipy.signal.welch)
    :param psd: 1D array of power spectral density values, same length as freqs
    :return: Peak frequency (Hz) as a float. Returns 0.0 if the spectrum has
        no power (e.g. an all-zero signal).
    """
    freqs = np.asarray(freqs)
    psd = np.asarray(psd)

    if psd.size == 0 or not np.any(psd):
        return 0.0

    return float(freqs[np.argmax(psd)])


def compute_spectral_frequency_metrics(signal, fs, nperseg=None):
    """
    Compute the median frequency (Median_Frequency) and peak frequency (Peak_Frequency) of a 1D signal
    from its Welch power spectral density. Same PSD-estimation convention as
    compute_band_powers (same default nperseg), so Median_Frequency/Peak_Frequency and band powers are
    computed from directly comparable spectra.

    :param signal: 1D array-like signal
    :param fs: Sampling frequency (Hz)
    :param nperseg: Segment length passed to scipy.signal.welch. Defaults to
        min(150, len(signal)), matching compute_band_powers.
    :return: Tuple (metrics, freqs, psd) -- metrics is a dict with 'Median_Frequency' and
        'Peak_Frequency' keys (Hz); freqs/psd are the raw Welch output.
    """
    signal = np.asarray(signal, dtype=float)

    if nperseg is None:
        nperseg = len(signal)//3

    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)

    metrics = {
        "Median_Frequency": compute_median_frequency(freqs, psd),
        "Peak_Frequency": compute_peak_frequency(freqs, psd),
    }

    return metrics, freqs, psd


def get_band_powers_in_phase(
    df,
    phase_start_idx,
    phase_end_idx,
    signal_column,
    fs,
    band_edges=None,
    nperseg=None,
):
    """
    Compute band-power spectral metrics for one signal, within a given sample
    window (the muscle's onset up to the concentric phase end for HD, or up
    to its offset for HI).

    :param df: DataFrame containing the signal
    :param phase_start_idx: Start sample index of the window (the muscle's onset)
    :param phase_end_idx: End sample index of the window
    :param signal_column: Column name of the signal to analyze
    :param fs: Sampling frequency (Hz)
    :param band_edges: See compute_band_powers
    :param nperseg: See compute_band_powers
    :return: Dict of band powers, see compute_band_powers
    """
    phase_signal = df[signal_column].iloc[phase_start_idx: phase_end_idx + 1].to_numpy()

    band_powers, _, _ = compute_band_powers(phase_signal, fs, band_edges=band_edges, nperseg=nperseg)

    return band_powers


def get_spectral_frequency_metrics_in_phase(
    df,
    phase_start_idx,
    phase_end_idx,
    signal_column,
    fs,
    nperseg=None,
):
    """
    Compute the median frequency (Median_Frequency) and peak frequency (Peak_Frequency) for one
    signal, within a given sample window. Same window convention as
    get_band_powers_in_phase (the muscle's own onset up to the concentric
    phase end for HD, or up to its own offset for HI).

    :param df: DataFrame containing the signal
    :param phase_start_idx: Start sample index of the window (the muscle's onset)
    :param phase_end_idx: End sample index of the window
    :param signal_column: Column name of the signal to analyze
    :param fs: Sampling frequency (Hz)
    :param nperseg: See compute_spectral_frequency_metrics
    :return: Dict with 'Median_Frequency' and 'Peak_Frequency' keys (Hz), see compute_spectral_frequency_metrics
    """
    phase_signal = df[signal_column].iloc[phase_start_idx: phase_end_idx + 1].to_numpy()

    metrics, _, _ = compute_spectral_frequency_metrics(phase_signal, fs, nperseg=nperseg)

    return metrics


def _safe_ratio(numerator, denominator):
    """
    Divide numerator by denominator, returning NaN instead of raising or
    producing inf when the denominator is zero (or either value is missing).

    :param numerator: Numerator value
    :param denominator: Denominator value
    :return: numerator / denominator as a float, or np.nan if denominator is 0
    """
    if denominator is None or numerator is None or denominator == 0:
        return float("nan")
    return float(numerator) / float(denominator)


def _compute_band_power_info_for_condition(
    df_normalized,
    onset_df,
    fs,
    onset_extensor_column,
    onset_flexor_column,
    emg_extensor_column,
    emg_flexor_column,
    band_edges,
    nperseg,
    start_idx_column,
    end_idx_column,
    rep_at_peak_power=None,
    peak_rep_label="rep_peakpower",
):
    """
    Compute per-repetition band-power and spectral-frequency (Median_Frequency/Peak_Frequency)
    metrics for a single condition's onset DataFrame. If rep_at_peak_power
    is provided, the matching repetition is keyed as peak_rep_label (e.g.
    'rep_peakpower' for HD, same convention as in
    emg_scalogram.get_muscle_scalogram_info; 'rep_max' for HI); if None,
    all repetitions keep their plain numeric key.

    The analysis window boundaries are read from start_idx_column/
    end_idx_column, independently for each muscle's row: the muscle's own
    onset ('onset_idx') up to the concentric phase end
    ('concentric_end_idx') for HD, or up to its own offset ('offset_idx')
    for HI.

    In addition to band powers, this also computes the median frequency
    (Median_Frequency) and peak frequency (Peak_Frequency) of each muscle within the same window, and
    the flexor/extensor ratio of each (Median_Frequency_ratio_flexor_extensor,
    Peak_Frequency_ratio_flexor_extensor) -- a NaN ratio means the extensor value was 0.

    :param df_normalized: DataFrame containing the normalized filtered EMG signals
    :param onset_df: DataFrame output for the condition from detect_emg_onsets
    :param fs: Sampling frequency (Hz)
    :param onset_extensor_column: Muscle column name used for extensor onsets
    :param onset_flexor_column: Muscle column name used for flexor onsets
    :param emg_extensor_column: Column name of the normalized filtered extensor EMG
    :param emg_flexor_column: Column name of the normalized filtered flexor EMG
    :param band_edges: See compute_band_powers
    :param nperseg: See compute_band_powers
    :param start_idx_column: Column in onset_df with the window start index
    :param end_idx_column: Column in onset_df with the window end index
    :param rep_at_peak_power: Repetition number to key as peak_rep_label, or None
    :param peak_rep_label: Key to use for the repetition matching rep_at_peak_power
    :return: Dict keyed by repetition (int or peak_rep_label) with a metrics
        dict containing "<band>Hz_extensor"/"<band>Hz_flexor" power values,
        "Median_Frequency_extensor"/"Median_Frequency_flexor", "Peak_Frequency_extensor"/"Peak_Frequency_flexor", and
        "Median_Frequency_ratio_flexor_extensor"/"Peak_Frequency_ratio_flexor_extensor"
    """
    band_power_info = {}

    repetition_groups = onset_df.groupby("rep")

    for rep, rep_data in repetition_groups:
        ext_row = rep_data[rep_data["muscle"] == onset_extensor_column]
        flex_row = rep_data[rep_data["muscle"] == onset_flexor_column]

        if ext_row.empty or flex_row.empty:
            continue

        # Extract the window boundaries independently per muscle
        extensor_start_idx = ext_row[start_idx_column].values[0]
        extensor_end_idx = ext_row[end_idx_column].values[0]
        flexor_start_idx = flex_row[start_idx_column].values[0]
        flexor_end_idx = flex_row[end_idx_column].values[0]

        if pd.isna(extensor_start_idx) or pd.isna(extensor_end_idx) or \
                pd.isna(flexor_start_idx) or pd.isna(flexor_end_idx):
            continue

        extensor_start_idx = int(extensor_start_idx)
        extensor_end_idx = int(extensor_end_idx)
        flexor_start_idx = int(flexor_start_idx)
        flexor_end_idx = int(flexor_end_idx)

        extensor_band_powers = get_band_powers_in_phase(
            df_normalized,
            extensor_start_idx,
            extensor_end_idx,
            emg_extensor_column,
            fs,
            band_edges=band_edges,
            nperseg=nperseg,
        )
        flexor_band_powers = get_band_powers_in_phase(
            df_normalized,
            flexor_start_idx,
            flexor_end_idx,
            emg_flexor_column,
            fs,
            band_edges=band_edges,
            nperseg=nperseg,
        )

        extensor_spectral_metrics = get_spectral_frequency_metrics_in_phase(
            df_normalized,
            extensor_start_idx,
            extensor_end_idx,
            emg_extensor_column,
            fs,
            nperseg=nperseg,
        )
        flexor_spectral_metrics = get_spectral_frequency_metrics_in_phase(
            df_normalized,
            flexor_start_idx,
            flexor_end_idx,
            emg_flexor_column,
            fs,
            nperseg=nperseg,
        )

        if rep_at_peak_power is not None and rep == rep_at_peak_power:
            rep_key = peak_rep_label
        else:
            rep_key = rep

        metrics = {}
        for band_label, power_value in extensor_band_powers.items():
            metrics[f"{band_label}_extensor"] = power_value
        for band_label, power_value in flexor_band_powers.items():
            metrics[f"{band_label}_flexor"] = power_value

        metrics["Median_Frequency_extensor"] = extensor_spectral_metrics["Median_Frequency"]
        metrics["Median_Frequency_flexor"] = flexor_spectral_metrics["Median_Frequency"]
        metrics["Peak_Frequency_extensor"] = extensor_spectral_metrics["Peak_Frequency"]
        metrics["Peak_Frequency_flexor"] = flexor_spectral_metrics["Peak_Frequency"]

        metrics["Median_Frequency_ratio"] = _safe_ratio(
            flexor_spectral_metrics["Median_Frequency"], extensor_spectral_metrics["Median_Frequency"]
        )
        metrics["Peak_Frequency_ratio"] = _safe_ratio(
            flexor_spectral_metrics["Peak_Frequency"], extensor_spectral_metrics["Peak_Frequency"]
        )

        band_power_info[rep_key] = metrics

    return band_power_info



def get_muscle_power_spectral_density_info(
    subject_data_normalized,
    onset_df_hd,
    onset_df_hi,
    power_info,
    onset_extensor_column="EMG_Extensor_filt",
    onset_flexor_column="EMG_Flexor_filt",
    emg_extensor_column="EMG_Extensor_filt_norm",
    emg_flexor_column="EMG_Flexor_filt_norm",
    band_edges=None,
    nperseg=None,
):
    """
    Build the per-repetition band-power and spectral-frequency (Median_Frequency/Peak_Frequency)
    info for both HD and HI conditions using their respective onset
    DataFrames from detect_emg_onsets.

    For each repetition, computes the power spectral density (Welch method)
    of the extensor and flexor EMG within an analysis window, then:
    - integrates it into frequency bands spanning 0-500 Hz (see
      compute_band_powers)
    - derives the median frequency (Median_Frequency) and peak frequency (Peak_Frequency) from the
      same spectrum (see compute_spectral_frequency_metrics)
    - computes the flexor/extensor ratio of Median_Frequency and of Peak_Frequency

    The window always starts at the muscle's own onset ('onset_idx'); it
    ends at the concentric phase end ('concentric_end_idx') for HD, and at
    the muscle's own offset ('offset_idx') for HI (isometric condition, no
    concentric/eccentric phases) -- the same window convention used in
    emg_scalogram.get_muscle_scalogram_info.

    The spectrum is computed on the normalized filtered EMG signal
    ('EMG_Extensor_filt_norm' / 'EMG_Flexor_filt_norm' by default), not on
    the normalized envelope, for the same reason as in
    get_muscle_scalogram_info: the envelope's smoothing removes most of the
    spectral content being characterized.

    HD has a repetition of peak power (identified via power_info and keyed
    as 'rep_peakpower' in the output, matching the convention used in
    get_muscle_scalogram_info so the two outputs can be joined on
    'repetition'). HI has no power_info-based peak repetition; instead, its
    repetition of maximum value is identified from
    subject_data["HI_values"] (a dict keyed "H1", "H2", ... whose highest
    value marks the max repetition) and keyed as 'rep_max' in the output.
    All other HI repetitions keep their plain numeric key.

    :param subject_data_normalized: Dict output of normalize_emg containing 'data' and 'params' for both conditions ('HD' and 'HI' keys)
    :param onset_df_hd: DataFrame output for condition 'HD' from detect_emg_onsets
    :param onset_df_hi: DataFrame output for condition 'HI' from detect_emg_onsets
    :param power_info: DataFrame containing a 'rep_at_peak_power' column (HD only)
    :param fs: Sampling frequency (Hz). If None, extracted per-condition from subject_data_normalized
    :param onset_extensor_column: Muscle column name used for extensor onsets
    :param onset_flexor_column: Muscle column name used for flexor onsets
    :param emg_extensor_column: Column name of the normalized (filtered) extensor EMG
    :param emg_flexor_column: Column name of the normalized (filtered) flexor EMG
    :param band_edges: See compute_band_powers. Defaults to 50 Hz wide bands spanning 0-500 Hz
    :param nperseg: See compute_band_powers
    :return: Tuple (df_hd, df_hi) containing tidy band-power + Median_Frequency/Peak_Frequency DataFrames, one row per repetition
    """
    rep_at_peak_power = int(power_info["rep_at_peak_power"].iloc[0])
    rep_at_max_hi = get_max_repetition_hi(subject_data_normalized)

    band_power_info_hd = _compute_band_power_info_for_condition(
        subject_data_normalized["HD"]["data"],
        onset_df_hd,
        FS,
        onset_extensor_column,
        onset_flexor_column,
        emg_extensor_column,
        emg_flexor_column,
        band_edges,
        nperseg,
        start_idx_column="onset_idx",
        end_idx_column="concentric_end_idx",
        rep_at_peak_power=rep_at_peak_power,
        peak_rep_label="rep_peakpower",
    )

    band_power_info_hi = _compute_band_power_info_for_condition(
        subject_data_normalized["HI"]["data"],
        onset_df_hi,
        FS,
        onset_extensor_column,
        onset_flexor_column,
        emg_extensor_column,
        emg_flexor_column,
        band_edges,
        nperseg,
        start_idx_column="onset_idx",
        end_idx_column="offset_idx",
        rep_at_peak_power=rep_at_max_hi,
        peak_rep_label="rep_max",
    )

    df_hd = build_band_power_dataframe({"HD": band_power_info_hd}, condition="HD")
    df_hi = build_band_power_dataframe({"HI": band_power_info_hi}, condition="HI")

    return df_hd, df_hi


def build_band_power_dataframe(band_power_info, condition="HD"):
    """
    Flatten band_power_info into a tidy DataFrame with one row per repetition.

    :param band_power_info: Dict containing metric values per repetition key
    :param condition: Condition key inside band_power_info (default 'HD')
    :return: Tidy DataFrame with one row per repetition and one column per
        "<band>Hz_<muscle>" band power, plus Median_Frequency/Peak_Frequency and ratio columns
    """
    rows = []
    for rep_key, metrics in band_power_info[condition].items():
        row = {"repetition": rep_key}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)

    return df
