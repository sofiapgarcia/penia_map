import os

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, correlate, correlation_lags
from constants import FS
import matplotlib.pyplot as plt


def apply_saved_sync_indices(subject_data, subject_folder,
                             hd_key="HD", hd_emg_key="HD_EMG",
                             filename="sync_indices.xlsx"):
    """
    If a sync_indices.xlsx file already exists for this subject, use it to
    trim the HD and HD_EMG DataFrames directly, without running the
    interactive synchronization again.

    Expects an Excel file with columns: acquisition | start_idx | end_idx
    and exactly two rows, one for hd_key and one for hd_emg_key.

    :param subject_data: Dict containing subject_data[hd_key]["data"] and
        subject_data[hd_emg_key]["data"] DataFrames
    :param subject_folder: Path to the subject's folder, where
        sync_indices.xlsx is expected
    :param hd_key: Key of the HD entry in subject_data
    :param hd_emg_key: Key of the HD_EMG entry in subject_data
    :param filename: Name of the Excel file with the saved indices
    :return: subject_data with HD/HD_EMG trimmed, if the file was found;
        otherwise None (so the caller knows to run the synchronization)
    """
    sync_path = os.path.join(subject_folder, filename)

    if not os.path.isfile(sync_path):
        return subject_data

    indices_df = pd.read_excel(sync_path)
    indices_df = indices_df.set_index("acquisition")

    for key in (hd_key, hd_emg_key):
        if key not in indices_df.index:
            print(f"[warning] '{key}' not found in {sync_path} — ignoring saved indices.")
            return None

    start_hd = int(indices_df.loc[hd_key, "start_idx"])
    end_hd = int(indices_df.loc[hd_key, "end_idx"])
    start_hd_emg = int(indices_df.loc[hd_emg_key, "start_idx"])
    end_hd_emg = int(indices_df.loc[hd_emg_key, "end_idx"])

    df_hd = subject_data[hd_key]["data"].iloc[start_hd:end_hd].reset_index(drop=True)
    df_hd_emg = subject_data[hd_emg_key]["data"].iloc[start_hd_emg:end_hd_emg].reset_index(drop=True)


    subject_data[hd_key]["data"] = df_hd
    subject_data[hd_emg_key]["data"] = df_hd_emg

    print(f"Loaded existing synchronization indices from: {sync_path}")

    return subject_data

def synchronize_hd_acquisitions(subject_data,
                                column_hd,
                                column_hd_emg,
                                subject_folder=None,
                                hd_key="HD",
                                hd_emg_key="HD_EMG",
                                target_bpm=90,
                                fs=FS,
                                max_lag_seconds=None):
    """
    Synchronize HD and HD_EMG acquisitions using maximum cross-correlation
    over an interactively-validated synchronization window.

    A validation window is shown with the candidate peaks detected on both
    signals (HD on top, HD_EMG on bottom). Left-click adds a peak (snapped
    to the nearest local maximum); right-click removes the nearest peak.
    Close the window to confirm the peaks and continue.

    At the end, a validation plot of the aligned signals is shown, and
    (if subject_folder is given) an Excel file with the start/end indices
    used to trim each acquisition is saved in that folder.

    :param subject_data: Dict containing at least subject_data[hd_key]["data"]
        and subject_data[hd_emg_key]["data"] DataFrames
    :param column_hd: Column name of the sync signal in the HD DataFrame
    :param column_hd_emg: Column name of the sync signal in the HD_EMG DataFrame
    :param subject_folder: Path to the subject's folder, where the
        synchronization indices Excel file will be saved. If None, the
        Excel file is not saved.
    :param hd_key: Key of the HD entry in subject_data
    :param hd_emg_key: Key of the HD_EMG entry in subject_data
    :param target_bpm: Expected beats-per-minute of the sync movement
    :param fs: Sampling frequency in Hz
    :param max_lag_seconds: Optional cap on the searched lag range, in seconds
    :return: subject_data, with the HD and HD_EMG "data" entries truncated and aligned
    """

    if hd_key not in subject_data:
        print(f"[warning] '{hd_key}' acquisition not found. Skipping synchronization.")
        return subject_data

    if hd_emg_key not in subject_data:
        print(f"[warning] '{hd_emg_key}' acquisition not found. Skipping synchronization.")
        return subject_data

    df_hd = subject_data[hd_key]["data"]
    df_hd_emg = subject_data[hd_emg_key]["data"]

    # HD accelerometer axis is flipped relative to HD_EMG's convention
    df_hd = _invert_column(df_hd, column_hd)

    offset = _compute_alignment_offset(
        df_hd[column_hd].values,
        df_hd_emg[column_hd_emg].values,
        fs,
        target_bpm,
        max_lag_seconds,
    )
    print("Offset estimate (cross-correlation):", offset)

    df_hd, df_hd_emg, indices = _trim_to_offset(df_hd, df_hd_emg, offset)

    _plot_validation(df_hd, df_hd_emg, column_hd, column_hd_emg)

    if subject_folder is not None:
        _save_sync_indices(subject_folder, indices, hd_key, hd_emg_key)
    else:
        print("[warning] subject_folder not provided — synchronization indices not saved.")

    subject_data[hd_key]["data"] = df_hd
    subject_data[hd_emg_key]["data"] = df_hd_emg

    return subject_data


def _invert_column(df, column):
    """
    Flip the sign of a single column.
    :param df: DataFrame
    :param column: column to be inverted
    :return: DataFrame with inverted column
    """
    df = df.copy()
    df[column] = -df[column]
    return df


def _compute_alignment_offset(signal_hd, signal_hd_emg, fs, target_bpm, max_lag_seconds):
    """
    Estimate the global sample offset between the two full signals by
    letting the user validate the sync-movement peaks on both signals,
    restricting each one to the resulting window, then cross-correlating
    within those windows.

    :return: Offset in samples, in the full-signal index frame
    """
    peaks_hd = _initial_sync_peaks(signal_hd, fs, target_bpm)
    peaks_hd_emg = _initial_sync_peaks(signal_hd_emg, fs, target_bpm)

    peaks_hd, peaks_hd_emg = _interactive_validate_peaks(
        signal_hd, signal_hd_emg, fs, peaks_hd, peaks_hd_emg
    )

    segment_hd, start_hd_win = _window_from_peaks(signal_hd, peaks_hd, fs)
    segment_hd_emg, start_hd_emg_win = _window_from_peaks(signal_hd_emg, peaks_hd_emg, fs)

    local_offset = _estimate_offset_by_correlation(
        segment_hd, segment_hd_emg, fs, max_lag_seconds=max_lag_seconds
    )

    # translate the offset found inside the windows back to full-signal indices
    return start_hd_win - start_hd_emg_win + local_offset


def _find_candidate_peaks(signal, fs, target_bpm, prominence_factor):
    """
    Find peaks likely to belong to the periodic sync movement. These are
    only the initial candidates shown to the user for manual validation.
    """
    signal = np.asarray(signal, dtype=float)
    expected_period = 60.0 / target_bpm
    expected_samples = int(expected_period * fs)
    prominence = prominence_factor * np.std(signal)

    peaks, _ = find_peaks(
        signal,
        prominence=prominence,
        distance=int(0.8 * expected_samples)
    )
    return peaks


def _find_longest_consistent_run(peaks, fs, target_bpm, tolerance):
    """
    Find the longest run of consecutive peak-to-peak intervals that
    match the expected sync period, within the given tolerance.

    :return: List of interval indices belonging to the best run, or None
    """
    expected_period = 60.0 / target_bpm
    intervals = np.diff(peaks) / fs
    matches = np.abs(intervals - expected_period) < expected_period * tolerance

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

    return best_run if best_run else None


def _initial_sync_peaks(signal, fs, target_bpm, tolerance=0.15, prominence_factor=2.0):
    """
    Get the initial guess of peaks belonging to the rhythmic sync
    movement: among all prominent peaks in the signal, keep only the
    longest run whose peak-to-peak intervals are consistent with
    target_bpm. This is just a starting point — the user can still add
    or remove peaks interactively afterwards.

    :return: Array of peak indices (may be empty if nothing consistent is found)
    """
    peaks = _find_candidate_peaks(signal, fs, target_bpm, prominence_factor)

    if len(peaks) < 2:
        return peaks

    best_run = _find_longest_consistent_run(peaks, fs, target_bpm, tolerance)

    if best_run is None:
        print(
            "[warning] no rhythmic run consistent with "
            f"target_bpm={target_bpm} was found automatically — "
            "please add the sync peaks manually in the validation window."
        )
        return np.array([], dtype=int)

    first = best_run[0]
    last = best_run[-1] + 1
    return peaks[first:last + 1]


def _peaks_to_offsets(signal, peaks, fs):
    """
    Convert a list/array of peak indices into (time, amplitude) pairs,
    suitable for a scatter plot.
    """
    if peaks is None or len(peaks) == 0:
        return np.empty((0, 2))
    peaks_arr = np.asarray(peaks)
    return np.c_[peaks_arr / fs, np.asarray(signal)[peaks_arr]]


def _interactive_validate_peaks(signal_hd, signal_hd_emg, fs, peaks_hd, peaks_hd_emg,
                                snap_window_seconds=0.15):
    """
    Show HD (top) and HD_EMG (bottom) signals with their candidate sync
    peaks marked, and let the user correct them before they're used to
    define the synchronization window.

    Left-click near a signal:  add a peak, snapped to the nearest local
                                maximum within a small window around the click.
    Right-click near a signal: remove the closest existing peak.
    Close the figure:          confirm and continue with the current peaks.

    :return: (peaks_hd, peaks_hd_emg) as sorted numpy arrays of indices
    """
    peaks_hd = list(np.asarray(peaks_hd).tolist())
    peaks_hd_emg = list(np.asarray(peaks_hd_emg).tolist())

    signal_hd = np.asarray(signal_hd, dtype=float)
    signal_hd_emg = np.asarray(signal_hd_emg, dtype=float)

    t_hd = np.arange(len(signal_hd)) / fs
    t_emg = np.arange(len(signal_hd_emg)) / fs

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    ax_top.plot(t_hd, signal_hd, color="tab:blue", label="HD", linewidth=0.8)
    scatter_top = ax_top.scatter(*_peaks_to_offsets(signal_hd, peaks_hd, fs).T
                                  if len(peaks_hd) else ([], []),
                                  color="red", zorder=5)
    ax_top.set_title("HD  —  left-click: add peak   |   right-click: remove nearest peak")
    ax_top.legend(loc="upper right")

    ax_bottom.plot(t_emg, signal_hd_emg, color="tab:orange", label="HD_EMG", linewidth=0.8)
    scatter_bottom = ax_bottom.scatter(*_peaks_to_offsets(signal_hd_emg, peaks_hd_emg, fs).T
                                        if len(peaks_hd_emg) else ([], []),
                                        color="red", zorder=5)
    ax_bottom.set_title("HD_EMG  —  left-click: add peak   |   right-click: remove nearest peak")
    ax_bottom.legend(loc="upper right")

    fig.suptitle("Validate synchronization peaks — close the window when done")

    def _snap_to_local_max(signal, click_x_seconds):
        idx = int(round(click_x_seconds * fs))
        half = int(snap_window_seconds * fs)
        lo = max(0, idx - half)
        hi = min(len(signal), idx + half)
        if hi <= lo:
            return int(np.clip(idx, 0, len(signal) - 1))
        return lo + int(np.argmax(signal[lo:hi]))

    def _nearest_peak_position(peaks, click_x_seconds):
        if len(peaks) == 0:
            return None
        peaks_arr = np.array(peaks)
        distances = np.abs(peaks_arr / fs - click_x_seconds)
        return int(np.argmin(distances))

    def _redraw():
        offsets_top = _peaks_to_offsets(signal_hd, peaks_hd, fs)
        offsets_bottom = _peaks_to_offsets(signal_hd_emg, peaks_hd_emg, fs)
        scatter_top.set_offsets(offsets_top)
        scatter_bottom.set_offsets(offsets_bottom)
        fig.canvas.draw_idle()

    def _on_click(event):
        if event.inaxes not in (ax_top, ax_bottom) or event.xdata is None:
            return

        if event.inaxes is ax_top:
            signal, peaks = signal_hd, peaks_hd
        else:
            signal, peaks = signal_hd_emg, peaks_hd_emg

        if event.button == 1:  # left click -> add peak
            new_peak = _snap_to_local_max(signal, event.xdata)
            if new_peak not in peaks:
                peaks.append(new_peak)
                peaks.sort()
        elif event.button == 3:  # right click -> remove nearest peak
            pos = _nearest_peak_position(peaks, event.xdata)
            if pos is not None:
                peaks.pop(pos)

        _redraw()

    cid = fig.canvas.mpl_connect("button_press_event", _on_click)
    plt.tight_layout()
    plt.show()
    fig.canvas.mpl_disconnect(cid)

    return np.array(sorted(peaks_hd), dtype=int), np.array(sorted(peaks_hd_emg), dtype=int)


def _window_from_peaks(signal, peaks, fs, margin_seconds=0.5):
    """
    Build the synchronization segment from the (user-validated) peaks:
    from the first to the last peak, plus a margin on each side.

    :return: Tuple (segment, start_idx) where segment is the signal slice used
    """
    if peaks is None or len(peaks) == 0:
        print("[warning] no synchronization peaks selected — using the whole signal instead.")
        return signal, 0

    margin = int(margin_seconds * fs)
    start_idx = max(0, int(peaks[0]) - margin)
    end_idx = min(len(signal), int(peaks[-1]) + margin)
    return signal[start_idx:end_idx], start_idx


def _estimate_offset_by_correlation(signal_a, signal_b, fs, max_lag_seconds=None):
    """
    Estimate the alignment offset between two signals via maximum
    cross-correlation.

    :param signal_a: Reference signal (e.g. HD)
    :param signal_b: Signal to align to the reference (e.g. HD_EMG)
    :param fs: Sampling frequency in Hz
    :param max_lag_seconds: Optional cap on the searched lag range, in seconds
    :return: Integer offset, in samples, such that signal_b shifted by offset aligns with signal_a
    """
    # z-score both signals so amplitude differences don't bias the correlation
    a = (signal_a - np.mean(signal_a)) / np.std(signal_a)
    b = (signal_b - np.mean(signal_b)) / np.std(signal_b)

    correlation = correlate(b, a, mode="full", method="fft")
    lags = correlation_lags(len(b), len(a), mode="full")

    if max_lag_seconds is not None:
        # ignore lags outside the physically plausible range
        max_lag_samples = int(max_lag_seconds * fs)
        valid = np.abs(lags) <= max_lag_samples
        correlation = correlation[valid]
        lags = lags[valid]

    best_lag = int(lags[np.argmax(correlation)])
    return best_lag


def _trim_to_offset(df_hd, df_hd_emg, offset):
    """
    Trim both DataFrames so they start at the same instant and have
    equal length, given the offset between them.

    offset is defined such that signal_hd_emg[i] aligns with
    signal_hd[i + offset]. A positive offset means HD needs to be
    trimmed from its start to line up with HD_EMG; a negative offset
    means HD_EMG needs to be trimmed instead.

    :return: (df_hd_trimmed, df_hd_emg_trimmed, indices), where indices is a
        dict with the start/end indices (in the ORIGINAL, untrimmed frame)
        used for each acquisition.
    """
    if offset >= 0:
        start_hd, start_hd_emg = offset, 0
    else:
        start_hd, start_hd_emg = 0, -offset

    n = min(len(df_hd) - start_hd, len(df_hd_emg) - start_hd_emg)

    end_hd = start_hd + n
    end_hd_emg = start_hd_emg + n

    df_hd = df_hd.iloc[start_hd:end_hd].reset_index(drop=True)
    df_hd_emg = df_hd_emg.iloc[start_hd_emg:end_hd_emg].reset_index(drop=True)

    indices = {
        "HD_start_idx": start_hd,
        "HD_end_idx": end_hd,
        "HD_EMG_start_idx": start_hd_emg,
        "HD_EMG_end_idx": end_hd_emg,
    }

    return df_hd, df_hd_emg, indices


def _save_sync_indices(subject_folder, indices, hd_key, hd_emg_key):
    """
    Save the start/end indices used to trim each acquisition into an
    Excel file inside the subject's folder.
    """
    os.makedirs(subject_folder, exist_ok=True)

    df = pd.DataFrame([
        {"acquisition": hd_key,
         "start_idx": indices["HD_start_idx"],
         "end_idx": indices["HD_end_idx"]},
        {"acquisition": hd_emg_key,
         "start_idx": indices["HD_EMG_start_idx"],
         "end_idx": indices["HD_EMG_end_idx"]},
    ])

    out_path = os.path.join(subject_folder, "sync_indices.xlsx")
    df.to_excel(out_path, index=False)
    print(f"Synchronization indices saved to: {out_path}")


def _plot_validation(df_hd, df_hd_emg, column_hd, column_hd_emg):
    """
    Overlay both z-scored signals so the alignment can be checked visually.
    """
    plt.figure(figsize=(12, 4))

    hd = (df_hd[column_hd] - df_hd[column_hd].mean()) / df_hd[column_hd].std()
    emg = (df_hd_emg[column_hd_emg] - df_hd_emg[column_hd_emg].mean()) / df_hd_emg[column_hd_emg].std()

    plt.plot(hd.values, label="HD")
    plt.plot(emg.values, label="HD_EMG")

    plt.legend()
    plt.title("Signals after synchronization")

    plt.show()