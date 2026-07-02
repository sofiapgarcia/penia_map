import numpy as np
from scipy.signal import (
    correlate,
    correlation_lags,
    find_peaks,
)
from constants import FS


def _detect_60bpm_window(signal, fs, target_bpm=60,
                         tolerance=0.15,
                         min_cycles=4,
                         prominence_factor=2.0):
    """
    Detect the synchronization movement performed at a known tempo.

    The function identifies a sequence of peaks occurring at approximately
    the expected period and returns both the synchronization window and the
    first synchronization peak.

    :param signal: One-dimensional synchronization signal
    :param fs: Sampling rate (Hz)
    :param target_bpm: Expected movement frequency (beats per minute)
    :param tolerance: Relative tolerance accepted for the expected period
    :param min_cycles: Minimum number of consecutive cycles required
    :param prominence_factor: Peak prominence multiplier
    :return: Dictionary containing:
             - start_idx: Beginning of the synchronization window
             - end_idx: End of the synchronization window
             - first_peak: Index of the first synchronization peak
             - last_peak: Index of the last synchronization peak
    """

    signal = np.asarray(signal, dtype=float)

    expected_period = 60.0 / target_bpm
    expected_samples = int(expected_period * fs)

    prominence = prominence_factor * np.std(signal)

    peaks, _ = find_peaks(signal, prominence=prominence,distance=int(0.8 * expected_samples))

    if len(peaks) < min_cycles:
        return None

    peak_times = peaks / fs

    intervals = np.diff(peak_times)

    matches = np.abs(intervals - expected_period) < (
        expected_period * tolerance
    )

    best_run = []
    current_run = []

    for i, match in enumerate(matches):

        if match:
            current_run.append(i)

        else:

            if len(current_run) > len(best_run):
                best_run = current_run

            current_run = []

    if len(current_run) > len(best_run):
        best_run = current_run

    if len(best_run) < (min_cycles - 1):
        return None

    first_peak = peaks[best_run[0]]

    last_peak = peaks[best_run[-1] + 1]

    margin = int(0.5 * fs)

    return {
        "start_idx": max(0, first_peak - margin),
        "end_idx": min(len(signal), last_peak + margin),
        "first_peak": first_peak,
        "last_peak": last_peak
    }

def _estimate_offset(segment_hd, segment_hd_emg, fs=FS):
    """
    Estimate the temporal offset between two synchronization segments.

    :param segment_hd: Synchronization segment from the HD acquisition
    :param segment_hd_emg: Synchronization segment from the HD_EMG acquisition
    :param fs: Sampling rate (Hz)
    :return: Estimated temporal offset (seconds)
    """

    # Normalize both signals before cross-correlation
    hd_norm = (
        segment_hd - np.mean(segment_hd)
    ) / np.std(segment_hd)

    hd_emg_norm = (
        segment_hd_emg - np.mean(segment_hd_emg)
    ) / np.std(segment_hd_emg)

    # Compute cross-correlation
    corr = correlate(hd_norm, hd_emg_norm, mode="full")

    lags = correlation_lags(
        len(hd_norm),
        len(hd_emg_norm),
        mode="full"
    )

    lag_samples = lags[np.argmax(corr)]

    return lag_samples / fs

def synchronize_hd_acquisitions(df_hd,
                                df_hd_emg,
                                column_hd,
                                column_hd_emg,
                                target_bpm=60,
                                fs=FS):
    """
    Synchronize the HD and HD_EMG acquisitions.

    The synchronization movement is detected automatically, the temporal
    offset is estimated by cross-correlation and both acquisitions are
    trimmed from the first synchronization peak to produce synchronized
    DataFrames with equal duration.

    :param df_hd: HD DataFrame
    :param df_hd_emg: HD_EMG DataFrame
    :param column_hd: Synchronization column in the HD acquisition
    :param column_hd_emg: Synchronization column in the HD_EMG acquisition
    :param target_bpm: Expected synchronization tempo (beats per minute)
    :param fs: Sampling rate (Hz)
    :return: Tuple containing:
             - synchronized HD DataFrame
             - synchronized HD_EMG DataFrame
    """

    # Detect the synchronization movement in both acquisitions
    sync_hd = _detect_60bpm_window(
        df_hd[column_hd],
        fs,
        target_bpm
    )

    sync_hd_emg = _detect_60bpm_window(
        df_hd_emg[column_hd_emg],
        fs,
        target_bpm
    )

    if sync_hd is None or sync_hd_emg is None:

        raise ValueError(
            "Unable to detect the synchronization movement."
        )

    # Extract synchronization segments
    segment_hd = df_hd[column_hd].values[
        sync_hd["start_idx"]:sync_hd["end_idx"]
    ]

    segment_hd_emg = df_hd_emg[column_hd_emg].values[
        sync_hd_emg["start_idx"]:sync_hd_emg["end_idx"]
    ]

    # Estimate the temporal offset between acquisitions
    offset_seconds = _estimate_offset(
        segment_hd,
        segment_hd_emg,
        fs
    )

    # Retrieve the first synchronization peak detected in each acquisition
    start_hd = sync_hd["first_peak"]
    start_hd_emg = sync_hd_emg["first_peak"]

    # Convert the temporal offset into samples
    offset = int(round(offset_seconds * fs))

    # Shift the delayed acquisition
    if offset > 0:
        start_hd_emg += offset

    elif offset < 0:
        start_hd += abs(offset)

    # Trim both acquisitions from the synchronized starting point
    df_hd = df_hd.iloc[start_hd:].reset_index(drop=True)

    df_hd_emg = df_hd_emg.iloc[start_hd_emg:].reset_index(drop=True)

    # Crop both DataFrames to the common duration
    n = min(len(df_hd), len(df_hd_emg))

    df_hd = df_hd.iloc[:n].reset_index(drop=True)

    df_hd_emg = df_hd_emg.iloc[:n].reset_index(drop=True)

    return df_hd, df_hd_emg