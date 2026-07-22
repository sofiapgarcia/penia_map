import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

EMG_COLUMNS = ['EMG_Extensor_filt', 'EMG_Flexor_filt']


def _tkeo(x):
    """
    Apply the Teager-Kaiser Energy Operator (TKEO) to a 1D signal.

    :param x: 1D array with the raw signal.
    :return: 1D array with the TKEO output (same length as x, first/last sample set to 0).
    """
    # TKEO formula: psi[n] = x[n]^2 - x[n-1] * x[n+1]
    # The first and last samples cannot be computed (no neighbors on both sides), so they stay at 0.
    tkeo = np.zeros_like(x, dtype=float)
    tkeo[1:-1] = x[1:-1] ** 2 - x[:-2] * x[2:]
    return tkeo


def _emg_envelope(x, fs, sigma_ms):
    """
    Compute a smooth energy envelope from a raw EMG signal.

    :param x: 1D array with the raw EMG signal.
    :param fs: Sampling rate (Hz).
    :param sigma_ms: Standard deviation (ms) of the Gaussian kernel used for smoothing.
    :return: 1D array with the smoothed energy envelope (same length as x).
    """
    # Step 1: TKEO highlights abrupt transitions (muscle activation) and attenuates slow variations.
    tkeo = _tkeo(x)

    # Step 2: rectification (absolute value), since only the energy magnitude matters.
    rectified = np.abs(tkeo)

    # Step 2 (cont.): Gaussian smoothing to obtain a continuous and stable envelope.
    sigma_samples = (sigma_ms / 1000.0) * fs
    return gaussian_filter1d(rectified, sigma=sigma_samples)


def _const_sse(y):
    """
    Compute the sum of squared errors of the best constant fit (mean) to a signal segment.

    :param y: 1D array with the signal segment.
    :return: Sum of squared errors of the constant fit.
    """
    # Represents the "baseline noise" model: the segment is assumed to be a constant level.
    if y.size == 0:
        return 0.0
    return float(np.sum((y - y.mean()) ** 2))


def _line_sse(y):
    """
    Compute the sum of squared errors of the best linear fit (regression) to a signal segment.

    :param y: 1D array with the signal segment.
    :return: Sum of squared errors of the linear fit.
    """
    # Represents the "activation ramp" model: the segment is assumed to rise linearly.
    n = y.size
    if n < 2:
        return 0.0

    x = np.arange(n)
    x_mean, y_mean = x.mean(), y.mean()
    sxx = np.sum((x - x_mean) ** 2)

    # Degenerate case (should not normally happen since x is a range of consecutive integers).
    if sxx == 0:
        return _const_sse(y)

    # Ordinary least squares slope and intercept.
    slope = np.sum((x - x_mean) * (y - y_mean)) / sxx
    intercept = y_mean - slope * x_mean

    residuals = y - (slope * x + intercept)
    return float(np.sum(residuals ** 2))


def _find_onset_in_window(envelope_window, guard_samples, min_window_samples):
    """
    Find the sample index that best splits a window into a constant segment and a rising segment.

    :param envelope_window: 1D array with the envelope samples inside the analysis window.
    :param guard_samples: Number of samples excluded at both edges of the window.
    :param min_window_samples: Minimum number of samples required to attempt a detection.
    :return: Local index (relative to the window) of the onset, or None if the window
        is too short for a reliable estimate.
    """
    n = envelope_window.size

    # Not enough signal in the window to distinguish baseline noise from muscle activation.
    if n < min_window_samples:
        return None

    # Exclude "guard" samples at both edges to avoid spurious detections near the boundaries.
    lo = guard_samples
    hi = n - guard_samples
    if hi <= lo:
        return None

    # Exhaustive search: try every candidate k and keep the one with the lowest combined error.
    best_k, best_err = None, np.inf
    for k in range(lo, hi):
        err = _const_sse(envelope_window[:k]) + _line_sse(envelope_window[k:])
        if err < best_err:
            best_err = err
            best_k = k

    return best_k


def _rep_numbers_from_phase(phase_col):
    """
    Extract the sorted list of repetition numbers from the 'Phase' column.

    :param phase_col: 1D array with the 'Phase' column values (e.g. 'C1', 'E1', 'R', ...).
    :return: Sorted list of repetition numbers found (based on the 'C{rep}' labels).
    """
    reps = set()
    for label in np.unique(phase_col):
        if isinstance(label, str) and label.startswith('C'):
            reps.add(int(label[1:]))
    return sorted(reps)


def _detect_emg_onsets_single_condition(df, fs, sigma_ms, window_ms, guard_ms, min_window_ms, columns):
    """
    Detect the EMG onset for both muscles, for every repetition, within a single
    condition that has a 'Phase' column (e.g. HD, where concentric/eccentric phases
    are identifiable).

    :param df: DataFrame containing the 'Phase' column and the EMG columns to process.
    :param fs: Sampling rate (Hz).
    :param sigma_ms: Standard deviation (ms) of the Gaussian kernel used to smooth the
        TKEO-based envelope.
    :param window_ms: Backward analysis window (ms) before each concentric start in
        which the EMG onset is searched.
    :param guard_ms: Exclusion margin (ms) at the window edges to avoid spurious
        detections near the boundaries.
    :param min_window_ms: Minimum usable window length (ms) to ensure sufficient
        signal for a reliable onset estimation.
    :param columns: Names of the EMG columns to process.
    :return: List of dicts, one per (repetition, muscle), with the keys: 'rep',
        'muscle', 'onset_idx' (global index, or None if not found),
        'concentric_start_idx', 'onset_to_start_ms' (time between the onset and the
        concentric start, in ms).
    """
    # This function relies on the 'Phase' column created by detect_concentric_eccentric_phases.
    if 'Phase' not in df.columns:
        raise ValueError("df has no 'Phase' column -- run detect_concentric_eccentric_phases first.")

    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in the data.")

    # Convert parameters from milliseconds to samples, based on the sampling rate.
    window_samples = int(round((window_ms / 1000.0) * fs))
    guard_samples = int(round((guard_ms / 1000.0) * fs))
    min_window_samples = int(round((min_window_ms / 1000.0) * fs))

    phase_col = df['Phase'].to_numpy()
    reps = _rep_numbers_from_phase(phase_col)
    if not reps:
        raise ValueError("No repetitions ('C1', 'C2', ...) found in the 'Phase' column.")

    # Compute the energy envelope once per muscle, over the full signal, to avoid
    # edge artifacts that would appear if computing it on small isolated windows.
    envelopes = {
        col: _emg_envelope(df[col].to_numpy(), fs, sigma_ms)
        for col in columns
    }

    results = []
    for rep_num in reps:
        # First sample index labeled as the concentric phase of this repetition.
        rep_idx = np.flatnonzero(phase_col == f'C{rep_num}')
        concentric_start = int(rep_idx[0])

        # Backward window before the concentric start (clipped at the start of the signal).
        window_start = max(0, concentric_start - window_samples)

        for col in columns:
            envelope_window = envelopes[col][window_start:concentric_start]
            local_onset = _find_onset_in_window(envelope_window, guard_samples, min_window_samples)

            if local_onset is None:
                results.append({
                    'rep': rep_num,
                    'muscle': col,
                    'onset_idx': None,
                    'concentric_start_idx': concentric_start,
                    'onset_to_start_ms': None,
                })
                continue

            # Convert the local (window-relative) index back to a global sample index.
            onset_idx = window_start + local_onset
            onset_to_start_ms = (concentric_start - onset_idx) / fs * 1000.0

            results.append({
                'rep': rep_num,
                'muscle': col,
                'onset_idx': onset_idx,
                'concentric_start_idx': concentric_start,
                'onset_to_start_ms': onset_to_start_ms,
            })

    return results


def _robust_baseline_threshold(envelope, threshold_k):
    """
    Estimate a robust detection threshold for an envelope signal, based on the median
    and the median absolute deviation (MAD).

    :param envelope: 1D array with the (smoothed) energy envelope.
    :param threshold_k: Multiplier applied to the robust spread estimate.
    :return: Scalar threshold value.
    """
    # Median/MAD are used instead of mean/std because the envelope is dominated, in
    # terms of sample count, by the "quiet" periods between contractions -- the median
    # stays close to the true baseline even with a few large bursts of muscle activity.
    baseline = np.median(envelope)
    mad = np.median(np.abs(envelope - baseline))
    robust_std = mad * 1.4826  # scale factor that makes MAD comparable to a normal std
    return baseline + threshold_k * robust_std


def _detect_active_segments(envelope, fs, threshold, min_duration_ms, min_gap_ms):
    """
    Detect contiguous segments where the envelope stays above a threshold, merging
    segments that are close together and discarding segments that are too short.

    :param envelope: 1D array with the (smoothed) energy envelope.
    :param fs: Sampling rate (Hz).
    :param threshold: Amplitude threshold above which the signal is considered "active".
    :param min_duration_ms: Minimum duration (ms) for a segment to be kept as a real contraction.
    :param min_gap_ms: Maximum gap (ms) between two active segments for them to be merged into one.
    :return: List of (start_idx, end_idx) tuples (end exclusive), sorted by start_idx.
    """
    above = envelope > threshold
    diff = np.diff(above.astype(int))
    starts = list(np.flatnonzero(diff == 1) + 1)
    ends = list(np.flatnonzero(diff == -1) + 1)
    if above.size and above[0]:
        starts = [0] + starts
    if above.size and above[-1]:
        ends = ends + [above.size]

    segments = list(zip(starts, ends))
    if not segments:
        return []

    # Merge segments separated by a gap shorter than min_gap_ms (bridges brief dips
    # within what is really a single, sustained isometric contraction).
    min_gap_samples = (min_gap_ms / 1000.0) * fs
    merged = [segments[0]]
    for s, e in segments[1:]:
        prev_s, prev_e = merged[-1]
        if s - prev_e < min_gap_samples:
            merged[-1] = (prev_s, e)
        else:
            merged.append((s, e))

    # Discard segments that are too short to be a real contraction (noise/artifacts).
    min_duration_samples = (min_duration_ms / 1000.0) * fs
    merged = [(s, e) for s, e in merged if (e - s) >= min_duration_samples]

    return merged


def _detect_emg_onsets_isometric_condition(df, fs, sigma_ms, window_ms, guard_ms, min_window_ms,
                                            columns, n_reps, threshold_k, min_duration_ms,
                                            min_gap_ms, forward_margin_ms):
    """
    Detect EMG onsets for isometric conditions (e.g. HI), where there is no 'Phase'
    column to mark a concentric start, since there's no concentric phase at all.

    Instead, a reference point per repetition is estimated directly from the envelope:
    each of the `n_reps` isometric contractions shows up as a period where the
    envelope stays above a robust, data-driven threshold (a "plateau"). The start of
    each plateau is used as the reference point, and the onset is then refined exactly
    like in the phase-based detection: by searching backward from that reference for
    the sample that best splits a constant baseline segment from a rising segment.

    :param df: DataFrame containing the EMG columns to process.
    :param fs: Sampling rate (Hz).
    :param sigma_ms: Standard deviation (ms) of the Gaussian kernel used to smooth the
        TKEO-based envelope.
    :param window_ms: Backward analysis window (ms) before each estimated reference
        point in which the EMG onset is searched.
    :param guard_ms: Exclusion margin (ms) at the window edges to avoid spurious
        detections near the boundaries.
    :param min_window_ms: Minimum usable window length (ms) to ensure sufficient
        signal for a reliable onset estimation.
    :param columns: Names of the EMG columns to process.
    :param n_reps: Expected number of isometric contractions (repetitions).
    :param threshold_k: Multiplier applied to the robust baseline spread (MAD-based) to
        set the "active" detection threshold. Increase if too many spurious segments
        are detected; decrease if real contractions are being missed.
    :param min_duration_ms: Minimum duration (ms) for a detected active segment to be
        accepted as a real contraction (filters out brief noise spikes).
    :param min_gap_ms: Maximum gap (ms) between two active segments for them to be
        merged into a single contraction (bridges short dips within a contraction).
    :param forward_margin_ms: Extra margin (ms) added past the detected threshold
        crossing, since the true physiological onset can occur slightly before the
        point where the envelope actually crosses the threshold.
    :return: List of dicts, one per (repetition, muscle), with the same keys as
        _detect_emg_onsets_single_condition's output. Note: 'concentric_start_idx'
        here holds the estimated reference point (plateau start + forward margin),
        not a true concentric start.
    """
    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in the data.")

    window_samples = int(round((window_ms / 1000.0) * fs))
    guard_samples = int(round((guard_ms / 1000.0) * fs))
    min_window_samples = int(round((min_window_ms / 1000.0) * fs))
    forward_margin_samples = int(round((forward_margin_ms / 1000.0) * fs))

    # Compute the energy envelope once per muscle, over the full signal.
    envelopes = {col: _emg_envelope(df[col].to_numpy(), fs, sigma_ms) for col in columns}

    results = []
    for col in columns:
        envelope = envelopes[col]
        threshold = _robust_baseline_threshold(envelope, threshold_k)
        segments = _detect_active_segments(envelope, fs, threshold, min_duration_ms, min_gap_ms)

        if len(segments) != n_reps:
            print(f"Warning: expected {n_reps} contractions for '{col}', found {len(segments)}. "
                  f"Adjust threshold_k / min_duration_ms / min_gap_ms if this looks wrong. "
                  f"Keeping the {n_reps} longest segments found, in temporal order.")
            segments = sorted(
                sorted(segments, key=lambda se: se[1] - se[0], reverse=True)[:n_reps],
                key=lambda se: se[0],
            )

        for rep_num, (seg_start, _seg_end) in enumerate(segments, start=1):
            reference_idx = min(seg_start + forward_margin_samples, envelope.size - 1)
            window_start = max(0, reference_idx - window_samples)

            envelope_window = envelope[window_start:reference_idx]
            local_onset = _find_onset_in_window(envelope_window, guard_samples, min_window_samples)

            if local_onset is None:
                results.append({
                    'rep': rep_num,
                    'muscle': col,
                    'onset_idx': None,
                    'concentric_start_idx': reference_idx,
                    'onset_to_start_ms': None,
                })
                continue

            onset_idx = window_start + local_onset
            onset_to_start_ms = (reference_idx - onset_idx) / fs * 1000.0

            results.append({
                'rep': rep_num,
                'muscle': col,
                'onset_idx': onset_idx,
                'concentric_start_idx': reference_idx,
                'onset_to_start_ms': onset_to_start_ms,
            })

    return results


def detect_emg_onsets(subject_data, conditions=('HD', 'HI'), sigma_ms=50, window_ms=300, guard_ms=10,
                       min_window_ms=100, emg_columns=None, show_plot=True,
                       n_reps=3, threshold_k=4.0, min_duration_ms=300, min_gap_ms=300,
                       forward_margin_ms=50, show_pipeline_plots=False,
                       pipeline_margin_before_ms=300, pipeline_margin_after_ms=200,
                       pipeline_max_reps=None):
    """
    Detect the EMG onset (start of muscle activation) for both muscles, for every
    repetition, in every requested condition, using a simplified and deterministic
    version of the EstOpt algorithm.

    Two detection paths are used automatically, depending on whether the condition's
    DataFrame has a 'Phase' column:
      - If 'Phase' is present (e.g. HD, where concentric/eccentric phases were
        identified), the concentric start of each repetition is used as the reference
        point (see _detect_emg_onsets_single_condition).
      - If 'Phase' is absent (e.g. HI, isometric contractions with no concentric
        phase), the reference point per repetition is estimated directly from the
        envelope by detecting the `n_reps` sustained "plateaus" above a robust
        threshold (see _detect_emg_onsets_isometric_condition).
    In both cases, the actual onset is refined the same way: by searching backward
    from the reference point for the sample that best separates a constant baseline
    segment from a rising segment.

    :param subject_data: Dict as produced/consumed by detect_concentric_eccentric_phases.
        For each condition in `conditions`, subject_data[condition]['data'] must contain
        the EMG columns (plus 'Phase' when available), and
        subject_data[condition]['params']['sampling rate'] the sampling frequency.
    :param conditions: Iterable with the condition names to process (e.g. 'HD', 'HI').
        Conditions missing from subject_data are skipped with a warning.
    :param sigma_ms: Standard deviation (ms) of the Gaussian kernel used to smooth the
        TKEO-based envelope.
    :param window_ms: Backward analysis window (ms) before each reference point in
        which the EMG onset is searched.
    :param guard_ms: Exclusion margin (ms) at the window edges to avoid spurious
        detections near the boundaries.
    :param min_window_ms: Minimum usable window length (ms) to ensure sufficient
        signal for a reliable onset estimation.
    :param emg_columns: Names of the EMG columns to process. Defaults to EMG_COLUMNS.
    :param show_plot: If True, plots both EMG signals with the detected onsets, per condition.
    :param n_reps: Expected number of isometric contractions per muscle, used only for
        conditions without a 'Phase' column (e.g. HI).
    :param threshold_k: Robust threshold multiplier used only for conditions without a
        'Phase' column.
    :param min_duration_ms: Minimum contraction duration (ms) used only for conditions
        without a 'Phase' column.
    :param min_gap_ms: Maximum gap (ms) to merge nearby active segments, used only for
        conditions without a 'Phase' column.
    :param forward_margin_ms: Margin (ms) added past the detected threshold crossing,
        used only for conditions without a 'Phase' column.
    :param show_pipeline_plots: If True, additionally plots -- for every repetition and
        muscle -- the intermediate signal-processing steps (raw EMG, TKEO, and
        |TKEO| overlaid with the smoothed envelope), with the reference point and the
        detected onset marked, for visual sanity-checking of the pipeline.
    :param pipeline_margin_before_ms: How much signal (ms) to show before the onset in
        the pipeline plots.
    :param pipeline_margin_after_ms: How much signal (ms) to show after the reference
        point in the pipeline plots.
    :param pipeline_max_reps: If set, limits the number of repetitions plotted per
        muscle in the pipeline plots (useful to avoid opening dozens of figures).
    :return: Dict keyed by condition name. Each value is a list of dicts, one per
        (repetition, muscle), with the keys: 'condition', 'rep', 'muscle', 'onset_idx'
        (global index, or None if not found), 'concentric_start_idx',
        'onset_to_start_ms' (time between the onset and the reference point, in ms).
    """
    columns = emg_columns if emg_columns is not None else EMG_COLUMNS

    results_by_condition = {}
    for condition in conditions:
        if condition not in subject_data:
            print(f"Warning: condition '{condition}' does not exist in subject_data -- skipping.")
            continue

        df = subject_data[condition]['data']
        fs = subject_data[condition]['params']['sampling rate']

        if 'Phase' in df.columns:
            condition_results = _detect_emg_onsets_single_condition(
                df, fs, sigma_ms, window_ms, guard_ms, min_window_ms, columns
            )
        else:
            condition_results = _detect_emg_onsets_isometric_condition(
                df, fs, sigma_ms, window_ms, guard_ms, min_window_ms, columns,
                n_reps, threshold_k, min_duration_ms, min_gap_ms, forward_margin_ms
            )

        # Tag every result with its condition, so results can be told apart once merged.
        for r in condition_results:
            r['condition'] = condition

        results_by_condition[condition] = condition_results

        if show_plot:
            _plot_onsets(df, columns, condition_results, condition)

        if show_pipeline_plots:
            _plot_pipeline_steps(
                df, columns, fs, sigma_ms, condition_results, condition,
                margin_before_ms=pipeline_margin_before_ms,
                margin_after_ms=pipeline_margin_after_ms,
                max_reps=pipeline_max_reps,
            )

    return results_by_condition


def _plot_pipeline_steps(df, columns, fs, sigma_ms, results, condition,
                          margin_before_ms=300, margin_after_ms=200, max_reps=None):
    """
    Plot, for every repetition and muscle, the intermediate signal-processing steps
    used to build the envelope and detect the EMG onset: the raw (filtered) EMG
    signal, the TKEO output, and the rectified TKEO (|TKEO|) overlaid with the
    smoothed envelope -- with the reference point and the detected onset marked on
    every panel, for visual sanity-checking of the pipeline.

    :param df: DataFrame containing the EMG columns to plot.
    :param columns: List of EMG column names to plot.
    :param fs: Sampling rate (Hz).
    :param sigma_ms: Standard deviation (ms) of the Gaussian kernel used to smooth the
        TKEO-based envelope (must match the value used for detection, so the plotted
        envelope is the same one used to find the onset).
    :param results: List of onset result dicts, as returned by
        _detect_emg_onsets_single_condition or _detect_emg_onsets_isometric_condition.
    :param condition: Name of the condition being plotted (used in the plot title).
    :param margin_before_ms: How much signal (ms) to show before the onset.
    :param margin_after_ms: How much signal (ms) to show after the reference point.
    :param max_reps: If set, limits the number of repetitions plotted per muscle.
    :return: None. Displays one figure per (muscle, repetition).
    """
    margin_before_samples = int(round((margin_before_ms / 1000.0) * fs))
    margin_after_samples = int(round((margin_after_ms / 1000.0) * fs))

    for col in columns:
        raw = df[col].to_numpy()

        # Recompute the same intermediate steps used during detection, so what's
        # plotted here matches exactly what the algorithm actually saw.
        tkeo = _tkeo(raw)
        rectified = np.abs(tkeo)
        sigma_samples = (sigma_ms / 1000.0) * fs
        envelope = gaussian_filter1d(rectified, sigma=sigma_samples)

        muscle_results = [r for r in results if r['muscle'] == col]
        if max_reps is not None:
            muscle_results = muscle_results[:max_reps]

        for r in muscle_results:
            ref_idx = r['concentric_start_idx']
            onset_idx = r['onset_idx']

            # Anchor the plotting window on the onset when available, otherwise on
            # the reference point (e.g. if no onset could be detected in that window).
            anchor = onset_idx if onset_idx is not None else ref_idx
            plot_start = max(0, anchor - margin_before_samples)
            plot_end = min(raw.size, ref_idx + margin_after_samples)
            sl = slice(plot_start, plot_end)
            t = np.arange(plot_start, plot_end)

            fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)

            axes[0].plot(t, raw[sl], color='black', linewidth=0.8)
            axes[0].set_ylabel('Raw EMG\n(filtered)', fontsize=8)

            axes[1].plot(t, tkeo[sl], color='tab:purple', linewidth=0.8)
            axes[1].set_ylabel('TKEO', fontsize=8)

            axes[2].plot(t, rectified[sl], color='gray', linewidth=0.6, alpha=0.6, label='|TKEO|')
            axes[2].plot(t, envelope[sl], color='tab:orange', linewidth=1.6, label='Smoothed envelope')
            axes[2].set_ylabel('Envelope', fontsize=8)
            axes[2].legend(fontsize=7, loc='upper left')

            for ax in axes:
                ax.axvline(ref_idx, color='tab:blue', linestyle='--', alpha=0.6)
                if onset_idx is not None:
                    ax.axvline(onset_idx, color='tab:red', linestyle='-', alpha=0.8)

            axes[-1].set_xlabel('Samples')
            fig.suptitle(
                f"{condition} -- {col} -- Rep {r['rep']} -- pipeline steps\n"
                f"(blue dashed = reference point, red = detected onset)",
                fontsize=10,
            )
            plt.tight_layout()
            plt.show()


def _plot_onsets(df, columns, results, condition):
    """
    Plot each EMG channel with the detected onsets and reference-point markers.

    :param df: DataFrame containing the EMG columns to plot.
    :param columns: List of EMG column names to plot.
    :param results: List of onset result dicts, as returned by
        _detect_emg_onsets_single_condition or _detect_emg_onsets_isometric_condition.
    :param condition: Name of the condition being plotted (used in the plot title).
    :return: None. Displays the plot.
    """
    n = len(columns)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.8 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        signal = df[col].to_numpy()
        ax.plot(signal, color='black', linewidth=0.8)

        # Only show each legend label once (for the first repetition plotted).
        col_results = [r for r in results if r['muscle'] == col]
        for i, r in enumerate(col_results):
            show_label = i == 0
            ax.axvline(r['concentric_start_idx'], color='tab:blue', linestyle='--', alpha=0.6,
                       label='Reference point' if show_label else None)
            if r['onset_idx'] is not None:
                ax.plot(r['onset_idx'], signal[r['onset_idx']], 'r^', markersize=8,
                       label='EMG onset' if show_label else None)

        ax.set_ylabel(col, fontsize=8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=8)
    axes[-1].set_xlabel('Samples')
    fig.suptitle(f'EMG Onset Detection -- {condition} Condition (all repetitions)')
    plt.tight_layout()
    plt.show()