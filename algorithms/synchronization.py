"""
Functions for synchronizing HD and HD_EMG acquisitions and correcting sample-rate drift.

Available Functions
-------------------
[Public]
apply_saved_sync_indices(...): Applies previously saved synchronization parameters to align and merge HD and HD_EMG acquisitions.
synchronize_hd_acquisitions(...): Synchronizes HD and HD_EMG signals using cross-correlation and interactive peak validation, manually refined alignment at two anchors, and linear drift correction.

[Private]
_invert_column(...): Inverts the sign of a specified DataFrame column.
_compute_alignment_offset(...): Estimates the initial sample offset between two signals using validated synchronization peaks and cross-correlation.
_find_candidate_peaks(...): Detects candidate peaks based on signal prominence and the expected movement period.
_find_longest_consistent_run(...): Identifies the longest sequence of consecutive peaks with intervals consistent with the expected movement period.
_initial_sync_peaks(...): Selects candidate synchronization peaks belonging to the longest rhythmically consistent sequence.
_peaks_to_offsets(...): Converts peak indices into time-amplitude coordinates for plotting.
_interactive_validate_peaks(...): Displays candidate synchronization peaks and allows interactive addition or removal before alignment.
_window_from_peaks(...): Extracts a signal segment spanning the selected peaks with a margin on each side.
_estimate_offset_by_correlation(...): Estimates the sample offset between two signals using cross-correlation.
_zscore(...): Standardizes a signal by subtracting its mean and dividing by its standard deviation.
_estimate_drift_correction(...): Estimates alignment offsets at the beginning and end of the recordings to determine the relative clock drift.
_interactive_local_alignment(...): Displays local signal windows and allows manual adjustment of the alignment offset using keyboard controls.
_apply_drift_correction(...): Resamples the HD signal onto the HD_EMG sample grid using linear interpolation to correct for clock drift.
_save_sync_params(...): Saves synchronization offsets and anchor indices to an Excel file.
_plot_validation(...): Plots the synchronized signals after drift correction to visually assess alignment.
"""
# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
from scipy.signal import find_peaks, correlate, correlation_lags
import matplotlib.pyplot as plt

# internal imports
from constants import FS

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def apply_saved_sync_indices(subject_data, subject_folder,
                             column_hd="gACCy",
                             hd_key="HD", hd_emg_key="HD_EMG",
                             filename="sync_indices.xlsx", show_plot=False):
    """
    If a sync_indices.xlsx file already exists for this subject, use the
    saved drift-correction parameters (offset_start, offset_end, anchors)
    to align and merge HD and HD_EMG without running the interactive
    synchronization again.

    Expects an Excel file with one row containing the columns:
    hd_key | hd_emg_key | offset_start | offset_end | anchor_start | anchor_end

    :param subject_data: Dict containing subject_data[hd_key]["data"] and
        subject_data[hd_emg_key]["data"] DataFrames
    :param subject_folder: Path to the subject's folder, where
        sync_indices.xlsx is expected
    :param hd_key: Key of the HD entry in subject_data
    :param hd_emg_key: Key of the HD_EMG entry in subject_data
    :param filename: Name of the Excel file with the saved parameters
    :return: subject_data with HD and HD_EMG aligned and merged into a
        single DataFrame under hd_key (hd_emg_key is removed), if the file
        was found; otherwise None (so the caller knows to run the
        synchronization)
    """
    sync_path = os.path.join(subject_folder, filename)

    if not os.path.isfile(sync_path):
        print("Sync indices were not found, retrieving original signals")
        return subject_data

    params_df = pd.read_excel(sync_path)

    match = params_df[
        (params_df["hd_key"] == hd_key) & (params_df["hd_emg_key"] == hd_emg_key)
    ]

    if match.empty:
        print(
            f"[warning] no saved parameters for '{hd_key}'/'{hd_emg_key}' "
            f"found in {sync_path} — ignoring saved parameters."
        )
        return None

    row = match.iloc[0]

    df_hd = subject_data[hd_key]["data"]
    df_hd_emg = subject_data[hd_emg_key]["data"]

    # HD accelerometer axis is flipped relative to HD_EMG's convention —
    # must match the inversion applied in synchronize_hd_acquisitions
    df_hd = _invert_column(df_hd, column_hd)

    df_hd_trim, df_hd_emg_trim, _ = _apply_drift_correction(
        df_hd, df_hd_emg,
        offset_start=int(row["offset_start"]),
        offset_end=int(row["offset_end"]),
        anchor_start=int(row["anchor_start"]),
        anchor_end=int(row["anchor_end"]),
    )

    df_merged = pd.concat([df_hd_trim, df_hd_emg_trim], axis=1)

    subject_data[hd_key]["data"] = df_merged
    del subject_data[hd_emg_key]

    print(f"Loaded existing synchronization parameters from: {sync_path}")

    if show_plot:
        _plot_validation(df_merged, column_hd="gACCy", column_hd_emg="y")

    return subject_data


def synchronize_hd_acquisitions(subject_data,
                                column_hd,
                                column_hd_emg,
                                subject_folder=None,
                                hd_key="HD",
                                hd_emg_key="HD_EMG",
                                target_bpm=90,
                                fs=FS,
                                max_lag_seconds=None,
                                anchor_margin_seconds=3.0,
                                local_window_seconds=6.0):
    """
    Synchronize HD and HD_EMG acquisitions, correcting not just a constant
    offset but also the sample-rate drift between the two recording
    systems (i.e. HD and HD_EMG don't run at exactly the same real
    sampling rate).

    Workflow:
      1. A validation window is shown with the candidate peaks detected on
         both signals (HD on top, HD_EMG on bottom). Left-click adds a peak
         (snapped to the nearest local maximum); right-click removes the
         nearest peak. Close the window to confirm the peaks and continue.
         This gives a coarse initial offset estimate via cross-correlation.
      2. A zoomed-in window near the validated sync peaks (start anchor)
         lets you fine-tune the offset there with the keyboard.
      3. A zoomed-in window near the END of the recording lets you
         fine-tune the offset there as well.
      4. The difference between the start and end offsets is used to
         estimate the drift rate between the two clocks, and HD_EMG is
         resampled (linearly, sample by sample) along the whole recording
         to compensate for it, so that both anchors (and everything in
         between) line up.
      5. A final validation plot is shown, and (if subject_folder is
         given) the offsets/anchors are saved to an Excel file so the same
         correction can be reapplied later without repeating the manual
         steps (see apply_saved_sync_indices).

    In steps 2 and 3, you can freely zoom/pan with the matplotlib toolbar;
    the keyboard shortcuts keep working no matter the zoom level:
        ->            shift the offset by +1 sample
        <-            shift the offset by -1 sample
        shift + ->    shift by +10 samples
        shift + <-    shift by -10 samples
        ctrl  + ->    shift by +100 samples
        ctrl  + <-    shift by -100 samples
        enter         confirm and close (closing the window also confirms)

    :param subject_data: Dict containing at least subject_data[hd_key]["data"]
        and subject_data[hd_emg_key]["data"] DataFrames
    :param column_hd: Column name of the sync signal in the HD DataFrame
    :param column_hd_emg: Column name of the sync signal in the HD_EMG DataFrame
    :param subject_folder: Path to the subject's folder, where the
        synchronization parameters Excel file will be saved. If None, the
        Excel file is not saved.
    :param hd_key: Key of the HD entry in subject_data
    :param hd_emg_key: Key of the HD_EMG entry in subject_data
    :param target_bpm: Expected beats-per-minute of the sync movement
        (used only for the initial coarse peak-based estimate)
    :param fs: Sampling frequency in Hz
    :param max_lag_seconds: Optional cap on the searched lag range for the
        coarse estimate, in seconds
    :param anchor_margin_seconds: For the start anchor, distance (in
        seconds) between the first validated sync peak and the anchor.
        For the end anchor, distance (in seconds) from the absolute end
        of the recording.
    :param local_window_seconds: Width (in seconds) of the zoomed-in
        comparison window shown around each anchor
    :return: subject_data, with the HD and HD_EMG "data" entries aligned
        (HD_EMG resampled to compensate for drift)
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

    signal_hd = df_hd[column_hd].values
    signal_hd_emg = df_hd_emg[column_hd_emg].values

    coarse_offset, peaks_hd_emg = _compute_alignment_offset(
        signal_hd, signal_hd_emg, fs, target_bpm, max_lag_seconds,
    )
    print("Coarse offset estimate (cross-correlation):", coarse_offset)

    offset_start, offset_end, anchor_start, anchor_end = _estimate_drift_correction(
        signal_hd, signal_hd_emg, fs, coarse_offset, peaks_hd_emg,
        anchor_margin_seconds, local_window_seconds,
    )
    print(f"Offset at start: {offset_start} samples | Offset at end: {offset_end} samples")

    df_hd_aligned, df_hd_emg_aligned, params = _apply_drift_correction(
        df_hd, df_hd_emg, offset_start, offset_end, anchor_start, anchor_end,
    )

    df_merged = pd.concat([df_hd_aligned, df_hd_emg_aligned], axis=1)

    _plot_validation(df_merged, column_hd, column_hd_emg)

    if subject_folder is not None:
        _save_sync_params(subject_folder, params, hd_key, hd_emg_key)
    else:
        print("[warning] subject_folder not provided — synchronization parameters not saved.")

    subject_data[hd_key]["data"] = df_merged
    del subject_data[hd_emg_key]

    return subject_data

# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

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
    Estimate a coarse global sample offset between the two full signals by
    letting the user validate the sync-movement peaks on both signals,
    restricting each one to the resulting window, then cross-correlating
    within those windows. This is only a starting point for the
    finer, drift-aware manual alignment that follows.

    :return: (offset, peaks_hd_emg) — offset in samples (full-signal index
        frame), and the user-validated HD_EMG sync peaks (also in the
        full-signal index frame), so the start anchor of the drift
        correction can be placed near them.
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
    offset = start_hd_win - start_hd_emg_win + local_offset

    return offset, peaks_hd_emg


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


def _initial_sync_peaks(signal, fs, target_bpm, tolerance=0.15, prominence_factor=4.0):
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
        # ignore clicks while a toolbar mode (zoom/pan) is active, so that
        # zooming doesn't accidentally add/remove peaks
        toolbar = fig.canvas.toolbar
        if toolbar is not None and toolbar.mode != "":
            return

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
    a = _zscore(signal_a)
    b = _zscore(signal_b)

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


def _zscore(x):
    """Z-score a 1D array (subtract mean, divide by std)."""
    x = np.asarray(x, dtype=float)
    std = np.std(x)
    if std == 0:
        return x - np.mean(x)
    return (x - np.mean(x)) / std


def _estimate_drift_correction(signal_hd, signal_hd_emg, fs, coarse_offset, peaks_hd_emg,
                                anchor_margin_seconds, local_window_seconds):
    """
    Let the user manually fine-tune the alignment offset near the
    validated sync peaks (start anchor) and near the end of the recording
    (end anchor), so the difference between the two can later be used to
    correct for clock drift between HD and HD_EMG.

    :param coarse_offset: Initial offset guess (e.g. from
        _compute_alignment_offset), used to initialize both passes
    :param peaks_hd_emg: Sync peaks validated on the HD_EMG signal (full-
        signal index frame). The start anchor is placed near the first of
        these peaks (+ anchor_margin_seconds), instead of near the
        absolute start of the recording. If empty, falls back to the
        absolute start of the recording.
    :param anchor_margin_seconds: For the start anchor, distance (seconds)
        between the first validated sync peak and the anchor. For the end
        anchor, distance (seconds) from the absolute end of the recording.
    :param local_window_seconds: Width (seconds) of the zoomed-in
        comparison window shown around each anchor
    :return: (offset_start, offset_end, anchor_start, anchor_end), all in
        samples, in the HD_EMG index frame for the anchors
    """
    n_emg = len(signal_hd_emg)
    margin = int(anchor_margin_seconds * fs)

    if peaks_hd_emg is not None and len(peaks_hd_emg) > 0:
        first_peak = int(peaks_hd_emg[0])
    else:
        print(
            "[warning] nenhum pico validado disponível — a usar o início "
            "absoluto do sinal como zona de referência para a âncora inicial."
        )
        first_peak = 0

    anchor_start = int(np.clip(first_peak + margin, 0, n_emg - 1))
    anchor_end = int(np.clip(n_emg - 1 - margin, 0, n_emg - 1))

    if anchor_end <= anchor_start:
        anchor_start, anchor_end = 0, n_emg - 1

    print("Alinhe o INÍCIO do sinal (use as setas do teclado; feche a janela para confirmar)...")
    offset_start = _interactive_local_alignment(
        signal_hd, signal_hd_emg, fs, anchor_start, coarse_offset,
        local_window_seconds, "início do sinal",
    )

    print("Alinhe o FIM do sinal (use as setas do teclado; feche a janela para confirmar)...")
    offset_end = _interactive_local_alignment(
        signal_hd, signal_hd_emg, fs, anchor_end, offset_start,
        local_window_seconds, "end do sinal",
    )

    return offset_start, offset_end, anchor_start, anchor_end


def _interactive_local_alignment(signal_hd, signal_hd_emg, fs, anchor_emg, initial_offset,
                                 window_seconds, label):
    """
    Show a zoomed-in window of both signals around a fixed anchor sample
    (in the HD_EMG index frame) and let the user nudge the local offset
    with the keyboard until they line up, at that specific point in time.

    The offset is defined such that signal_hd_emg[anchor_emg] aligns with
    signal_hd[anchor_emg + offset] (same convention used everywhere else
    in this module).

    You can freely zoom/pan with the matplotlib toolbar; the keyboard
    shortcuts keep working no matter the zoom level (each redraw resets
    the x-limits to the full local window, but you can zoom back in right
    after).

    Keyboard controls:
        ->            shift the offset by +1 sample
        <-            shift the offset by -1 sample
        shift + ->    shift by +10 samples
        shift + <-    shift by -10 samples
        ctrl  + ->    shift by +100 samples
        ctrl  + <-    shift by -100 samples
        enter         confirm and close

    Closing the window (by any means) also confirms the current offset.

    :return: final local offset (int)
    """
    half = max(1, int(window_seconds / 2 * fs))
    state = {"offset": int(initial_offset)}

    def _get_windows(offset):
        emg_lo = max(0, anchor_emg - half)
        emg_hi = min(len(signal_hd_emg), anchor_emg + half)
        hd_lo = emg_lo + offset
        hd_hi = emg_hi + offset

        # if the shifted HD window falls outside the signal, shrink both
        # windows symmetrically so they stay the same length and in range
        clip_lo = max(0, -hd_lo)
        clip_hi = max(0, hd_hi - len(signal_hd))
        hd_lo += clip_lo
        hd_hi -= clip_hi
        emg_lo += clip_lo
        emg_hi -= clip_hi

        if hd_hi <= hd_lo or emg_hi <= emg_lo:
            return np.array([]), np.array([]), np.array([])

        t = (np.arange(emg_lo, emg_hi) - anchor_emg) / fs
        hd_seg = _zscore(signal_hd[hd_lo:hd_hi])
        emg_seg = _zscore(signal_hd_emg[emg_lo:emg_hi])
        return t, hd_seg, emg_seg

    t, hd_seg, emg_seg = _get_windows(state["offset"])

    # Free up the left/right arrow keys, which matplotlib binds by default
    # to "back"/"forward" through the view history — we want them
    # exclusively for offset adjustment, even while zoomed in.
    prev_back = mpl.rcParams["keymap.back"]
    prev_forward = mpl.rcParams["keymap.forward"]
    mpl.rcParams["keymap.back"] = [k for k in prev_back if k not in ("left", "c", "backspace")]
    mpl.rcParams["keymap.forward"] = [k for k in prev_forward if k not in ("right", "v")]

    fig, ax = plt.subplots(figsize=(12, 4.5))
    line_hd, = ax.plot(t, hd_seg, label="HD", linewidth=0.9)
    line_emg, = ax.plot(t, emg_seg, label="HD_EMG", linewidth=0.9)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.legend(loc="upper right")
    title = ax.set_title("")

    def _update_title():
        title.set_text(
            f"Alinhamento manual — {label}  |  offset atual = {state['offset']} amostras "
            f"({state['offset'] / fs * 1000:.1f} ms)\n"
            "\u2190/\u2192: \u00b11 amostra   |   shift+\u2190/\u2192: \u00b110   |   "
            "ctrl+\u2190/\u2192: \u00b1100   |   pode dar zoom livremente — "
            "feche a janela (ou ENTER) para confirmar"
        )

    _update_title()

    def _redraw():
        t, hd_seg, emg_seg = _get_windows(state["offset"])
        line_hd.set_data(t, hd_seg)
        line_emg.set_data(t, emg_seg)
        if len(t) > 0:
            ax.set_xlim(t[0], t[-1])
        _update_title()
        fig.canvas.draw_idle()

    def _on_key(event):
        key = event.key
        if key is None:
            return

        if key in ("right", "shift+right", "ctrl+right"):
            step = 100 if key == "ctrl+right" else (10 if key == "shift+right" else 1)
            state["offset"] += step
            _redraw()
        elif key in ("left", "shift+left", "ctrl+left"):
            step = 100 if key == "ctrl+left" else (10 if key == "shift+left" else 1)
            state["offset"] -= step
            _redraw()
        elif key == "enter":
            plt.close(fig)

    cid = fig.canvas.mpl_connect("key_press_event", _on_key)
    plt.tight_layout()
    plt.show()
    fig.canvas.mpl_disconnect(cid)

    # restore global rcParams so we don't affect other plots in the session
    mpl.rcParams["keymap.back"] = prev_back
    mpl.rcParams["keymap.forward"] = prev_forward

    return state["offset"]


def _apply_drift_correction(df_hd, df_hd_emg, offset_start, offset_end, anchor_start, anchor_end):
    """
    Resample HD onto HD_EMG's sample grid, correcting for the linear
    clock drift implied by having two different (offset, anchor) pairs
    instead of a single constant offset.

    NOTE ON THIS VERSION: there is no physical reason to treat either
    acquisition system as the "stable" one — both clocks can drift
    relative to each other, and we have no way of knowing which one is
    "truer". So the choice of which signal gets interpolated (and
    therefore very slightly smoothed) and which one keeps its original
    samples untouched is arbitrary. This version keeps HD_EMG's original
    grid intact and interpolates HD onto it instead of the other way
    around. Because of this, no inversion of the offset(j) line is
    needed: the same forward equation used to define the drift model is
    directly usable as the interpolation grid.

    Model: for a HD_EMG sample at index j, the corresponding HD index is
        k(j) = j + offset(j)
    where offset(j) is a straight line through (anchor_start, offset_start)
    and (anchor_end, offset_end). For every integer HD_EMG index j in
    range, we compute the (generally fractional) HD index k(j) directly,
    and linearly interpolate HD's columns there.

    :return: (df_hd_resampled, df_hd_emg_trimmed, params), where params is
        a dict with the parameters needed to redo this exact correction
        later (see apply_saved_sync_indices)
    """
    n_hd = len(df_hd)
    n_emg = len(df_hd_emg)

    if anchor_end == anchor_start:
        slope = 0.0
    else:
        slope = (offset_end - offset_start) / (anchor_end - anchor_start)

    # forward equation: for every HD_EMG index j, where does it fall in HD?
    j_all = np.arange(n_emg)
    k_all = j_all * (1 + slope) + offset_start - slope * anchor_start

    # keep only the HD_EMG samples whose corresponding HD point actually
    # exists (i.e. falls inside [0, n_hd - 1])
    valid = (k_all >= 0) & (k_all <= n_hd - 1)

    if not valid.any():
        raise ValueError(
            "No valid overlap found between HD and HD_EMG with the given "
            "offsets — please redo the manual alignment."
        )

    j_min = int(np.where(valid)[0].min())
    j_max = int(np.where(valid)[0].max())

    j_grid = np.arange(j_min, j_max + 1)
    k_grid = j_grid * (1 + slope) + offset_start - slope * anchor_start
    k_grid = np.clip(k_grid, 0, n_hd - 1)

    df_hd_emg_trim = df_hd_emg.iloc[j_min:j_max + 1].reset_index(drop=True)

    k_source = np.arange(n_hd)
    resampled_cols = {}
    for col in df_hd.columns:
        values = df_hd[col].values
        if np.issubdtype(values.dtype, np.number):
            resampled_cols[col] = np.interp(k_grid, k_source, values)
        else:
            # non-numeric columns can't be interpolated: take the nearest
            # original sample instead
            nearest_idx = np.round(k_grid).astype(int)
            resampled_cols[col] = values[nearest_idx]

    df_hd_trim = pd.DataFrame(resampled_cols)

    params = {
        "offset_start": int(offset_start),
        "offset_end": int(offset_end),
        "anchor_start": int(anchor_start),
        "anchor_end": int(anchor_end),
    }

    return df_hd_trim, df_hd_emg_trim, params


def _save_sync_params(subject_folder, params, hd_key, hd_emg_key):
    """
    Save the drift-correction parameters (offsets and anchors) into an
    Excel file inside the subject's folder, so the same correction can be
    reapplied later via apply_saved_sync_indices.
    """
    os.makedirs(subject_folder, exist_ok=True)

    df = pd.DataFrame([{
        "hd_key": hd_key,
        "hd_emg_key": hd_emg_key,
        "offset_start": params["offset_start"],
        "offset_end": params["offset_end"],
        "anchor_start": params["anchor_start"],
        "anchor_end": params["anchor_end"],
    }])

    out_path = os.path.join(subject_folder, "sync_indices.xlsx")
    df.to_excel(out_path, index=False)
    print(f"Synchronization parameters saved to: {out_path}")


def _plot_validation(df_merged, column_hd, column_hd_emg):
    """
    Overlay both z-scored signals so the alignment can be checked visually,
    across the whole recording.
    """
    plt.figure(figsize=(12, 4))

    hd = _zscore(df_merged[column_hd].values)
    emg = _zscore(df_merged[column_hd_emg].values)

    plt.plot(hd, label="HD")
    plt.plot(emg, label="HD_EMG")

    plt.legend()
    plt.title("Signals after synchronization (drift-corrected)")

    plt.show()