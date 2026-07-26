import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

EMG_COLUMNS = ['EMG_Extensor_filt', 'EMG_Flexor_filt']


def _tkeo(x):
    """
    Apply the Teager-Kaiser Energy Operator (TKEO) to a 1D signal.

    :param x: 1D array with the raw signal.
    :return: 1D array with the TKEO output (same length as x, first/last sample set to 0).
    """
    tkeo = np.zeros_like(x, dtype=float)
    tkeo[1:-1] = x[1:-1] ** 2 - x[:-2] * x[2:]
    return tkeo


def _emg_envelope(x, fs, window_ms):
    """
    Compute a smooth energy envelope from a raw EMG signal using a moving average.

    :param x: 1D array with the raw EMG signal.
    :param fs: Sampling rate (Hz).
    :param window_ms: Moving average window length (ms).
    :return: 1D array with the smoothed energy envelope (same length as x).
    """
    rectified = np.abs(x)
    tkeo = _tkeo(rectified)
    tkeo_rectified = np.abs(tkeo)

    window_samples = max(1, int(round((window_ms / 1000.0) * fs)))
    kernel = np.ones(window_samples) / window_samples

    return np.convolve(tkeo_rectified, kernel, mode="same")


def _const_sse(y):
    """
    Compute the sum of squared errors of the best constant fit (mean) to a signal segment.

    :param y: 1D array with the signal segment.
    :return: Sum of squared errors of the constant fit.
    """
    if y.size == 0:
        return 0.0
    return float(np.sum((y - y.mean()) ** 2))


def _line_sse(y):
    """
    Compute the sum of squared errors of the best linear fit (regression) to a signal segment.

    :param y: 1D array with the signal segment.
    :return: Sum of squared errors of the linear fit.
    """
    n = y.size
    if n < 2:
        return 0.0

    x = np.arange(n)
    x_mean, y_mean = x.mean(), y.mean()
    sxx = np.sum((x - x_mean) ** 2)

    if sxx == 0:
        return _const_sse(y)

    slope = np.sum((x - x_mean) * (y - y_mean)) / sxx
    intercept = y_mean - slope * x_mean

    residuals = y - (slope * x + intercept)
    return float(np.sum(residuals ** 2))


def _find_onset_in_window(envelope_window, guard_samples, min_window_samples):
    """
    Find the sample index that best splits a window into a constant (baseline)
    segment followed by a rising (activation ramp) segment.

    :param envelope_window: 1D array with the envelope samples inside the analysis window.
    :param guard_samples: Number of samples excluded at both edges of the window.
    :param min_window_samples: Minimum number of samples required to attempt a detection.
    :return: Local index (relative to the window) of the onset, or None if the window
        is too short for a reliable estimate.
    """
    n = envelope_window.size
    if n < min_window_samples:
        return None

    lo = guard_samples
    hi = n - guard_samples
    if hi <= lo:
        return None

    best_k, best_err = None, np.inf
    for k in range(lo, hi):
        # constant segment BEFORE k (baseline), line segment AFTER k (ramp up)
        err = _const_sse(envelope_window[:k]) + _line_sse(envelope_window[k:])
        if err < best_err:
            best_err = err
            best_k = k

    return best_k


def _find_offset_in_window(envelope_window, guard_samples, min_window_samples):
    """
    Find the sample index that best splits a window into a falling (deactivation
    ramp) segment followed by a constant (post-contraction baseline) segment.

    This is the mirror image of _find_onset_in_window: instead of looking for
    "baseline -> ramp up", it looks for "ramp down -> baseline", which is the
    pattern expected at the END of a contraction.

    :param envelope_window: 1D array with the envelope samples inside the analysis window.
    :param guard_samples: Number of samples excluded at both edges of the window.
    :param min_window_samples: Minimum number of samples required to attempt a detection.
    :return: Local index (relative to the window) of the offset, or None if the window
        is too short for a reliable estimate.
    """
    n = envelope_window.size
    if n < min_window_samples:
        return None

    lo = guard_samples
    hi = n - guard_samples
    if hi <= lo:
        return None

    best_k, best_err = None, np.inf
    for k in range(lo, hi):
        # line segment BEFORE k (ramp down), constant segment AFTER k (baseline)
        err = _line_sse(envelope_window[:k]) + _const_sse(envelope_window[k:])
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
        'concentric_start_idx', 'concentric_end_idx', 'onset_to_start_ms'.
    """
    if 'Phase' not in df.columns:
        raise ValueError("df has no 'Phase' column -- run detect_concentric_eccentric_phases first.")

    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in the data.")

    window_samples = int(round((window_ms / 1000.0) * fs))
    guard_samples = int(round((guard_ms / 1000.0) * fs))
    min_window_samples = int(round((min_window_ms / 1000.0) * fs))

    phase_col = df['Phase'].to_numpy()
    reps = _rep_numbers_from_phase(phase_col)
    if not reps:
        raise ValueError("No repetitions ('C1', 'C2', ...) found in the 'Phase' column.")

    envelopes = {
        col: _emg_envelope(df[col].to_numpy(), fs, sigma_ms)
        for col in columns
    }

    results = []
    for rep_num in reps:
        rep_idx = np.flatnonzero(phase_col == f'C{rep_num}')
        concentric_start = int(rep_idx[0])
        concentric_end = int(rep_idx[-1])

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
                    'concentric_end_idx': concentric_end,
                    'onset_to_start_ms': None,
                })
                continue

            onset_idx = window_start + local_onset
            onset_to_start_ms = (concentric_start - onset_idx) / fs * 1000.0

            results.append({
                'rep': rep_num,
                'muscle': col,
                'onset_idx': onset_idx,
                'concentric_start_idx': concentric_start,
                'concentric_end_idx': concentric_end,
                'onset_to_start_ms': onset_to_start_ms,
            })

    return results


import numpy as np

def _robust_baseline_threshold(envelope, top_percentage=0.05, threshold_ratio=0.6):
    """
    Compute a threshold based on the median of the highest envelope values.

    :param envelope: 1D array with the (smoothed) energy envelope.
    :param top_percentage: Fraction of the highest values to consider
        (e.g. 0.1 = top 10%).
    :param threshold_ratio: Fraction of the median used as threshold.
    :return: Scalar threshold value.
    """
    envelope = np.asarray(envelope)

    # Number of samples to keep
    n_top = max(1, int(len(envelope) * top_percentage))
    print(n_top)

    # Highest values
    top_values = np.sort(envelope)[-n_top:]

    # Median of the highest values
    reference = np.median(top_values)
    print(reference)

    return reference * threshold_ratio


def _raw_active_segments(envelope, threshold):
    """
    Find every contiguous segment where the envelope is above the threshold,
    with NO merging and NO duration filtering applied yet. This is the "before"
    picture used by `_detect_active_segments` -- useful on its own for visualizing
    the coarse (unfiltered) detection step, e.g. in `_plot_hi_coarse_detection`.

    :param envelope: 1D array with the (smoothed) energy envelope.
    :param threshold: Amplitude threshold above which the signal is considered "active".
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
    return list(zip(starts, ends))


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
    segments = _raw_active_segments(envelope, threshold)
    if not segments:
        return []

    min_gap_samples = (min_gap_ms / 1000.0) * fs
    merged = [segments[0]]
    for s, e in segments[1:]:
        prev_s, prev_e = merged[-1]
        if s - prev_e < min_gap_samples:
            merged[-1] = (prev_s, e)
        else:
            merged.append((s, e))

    min_duration_samples = (min_duration_ms / 1000.0) * fs
    merged = [(s, e) for s, e in merged if (e - s) >= min_duration_samples]

    return merged


def _plot_hi_coarse_detection(df, columns, fs, sigma_ms, threshold_k, min_duration_ms, min_gap_ms, condition):
    """
    Plot, for every muscle in an isometric condition (e.g. HI), the three stages of
    the COARSE contraction detection step (before onset/offset refinement):
      1. Raw (filtered) EMG signal.
      2. Envelope + robust threshold + every raw threshold crossing (unfiltered).
      3. Final segments kept after merging close crossings (`min_gap_ms`) and
         discarding short ones (`min_duration_ms`).

    This is meant purely for visual sanity-checking of `threshold_k`, `min_gap_ms`
    and `min_duration_ms` -- it does NOT perform onset/offset refinement (see
    `_plot_hi_pipeline_steps` for that).

    :param df: DataFrame containing the EMG columns to plot.
    :param columns: List of EMG column names to plot (one figure per column).
    :param fs: Sampling rate (Hz).
    :param sigma_ms: Moving-average window (ms) used to build the envelope (must
        match the value used for detection).
    :param threshold_k: Multiplier applied to the robust baseline spread (MAD-based)
        to set the "active" detection threshold (must match the value used for
        detection).
    :param min_duration_ms: Minimum duration (ms) for a detected active segment to be
        accepted as a real contraction (must match the value used for detection).
    :param min_gap_ms: Maximum gap (ms) between two active segments for them to be
        merged into a single contraction (must match the value used for detection).
    :param condition: Name of the condition being plotted (used in the plot title).
    :return: None. Displays one figure per muscle column.
    """
    for col in columns:
        raw = df[col].to_numpy()
        t = np.arange(raw.size) / fs

        envelope = _emg_envelope(raw, fs, sigma_ms)
        threshold = _robust_baseline_threshold(envelope)
        raw_segments = _raw_active_segments(envelope, threshold)
        kept_segments = _detect_active_segments(envelope, fs, threshold, min_duration_ms, min_gap_ms)

        fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)

        axes[0].plot(t, raw, color='black', linewidth=0.6)
        axes[0].set_ylabel('Raw EMG\n(filtered)', fontsize=9)
        axes[0].set_title(f'{condition} -- {col} -- 1. Raw signal')

        axes[1].plot(t, envelope, color='tab:orange', linewidth=1.2, label='envelope')
        axes[1].axhline(threshold, color='tab:brown', linestyle=':', linewidth=1.5,
                         label=f'threshold = median + {threshold_k}xMAD')
        for i, (s, e) in enumerate(raw_segments):
            axes[1].axvspan(t[s], t[e - 1], color='tab:red', alpha=0.15,
                             label='raw segment (unfiltered)' if i == 0 else None)
        axes[1].set_ylabel('Envelope', fontsize=9)
        axes[1].legend(fontsize=8, loc='upper right')
        axes[1].set_title('2. Coarse detection: every raw threshold crossing')

        axes[2].plot(t, envelope, color='tab:orange', linewidth=1.2)
        axes[2].axhline(threshold, color='tab:brown', linestyle=':', linewidth=1.5)
        for i, (s, e) in enumerate(kept_segments):
            axes[2].axvspan(t[s], t[e - 1], color='tab:green', alpha=0.25,
                             label='kept segment (merged + duration-filtered)' if i == 0 else None)
        axes[2].set_ylabel('Envelope', fontsize=9)
        axes[2].set_xlabel('Time (s)')
        axes[2].legend(fontsize=8, loc='upper right')
        axes[2].set_title(
            f'3. Final segments: merged if gap < {min_gap_ms}ms, kept if duration >= {min_duration_ms}ms '
            f'-> {len(kept_segments)} segments found'
        )

        plt.tight_layout()
        plt.show()


def _detect_emg_onsets_isometric_condition(df, fs, sigma_ms, window_ms, guard_ms, min_window_ms,
                                            columns, n_reps, threshold_k, min_duration_ms,
                                            min_gap_ms, forward_margin_ms, offset_margin_ms=None):
    """
    Detect EMG onsets AND offsets for isometric conditions (e.g. HI), where there is
    no 'Phase' column to mark a concentric start/end, since there's no concentric
    (nor eccentric) phase at all -- only an activation onset and offset per repetition.

    A reference point per repetition is estimated directly from the envelope: each
    of the `n_reps` isometric contractions shows up as a period where the envelope
    stays above a robust, data-driven threshold (a "plateau").
      - The onset is refined by searching BACKWARD from the plateau start for the
        sample that best splits a constant baseline segment from a rising segment.
      - The offset is refined by searching FORWARD from the plateau end for the
        sample that best splits a falling (deactivation) segment from a constant
        post-contraction baseline segment -- the mirror image of the onset search.

    Because the envelope is computed independently per muscle, the plateau (and
    therefore both onset and offset) can fall at slightly different sample indices
    for each muscle. The plateau reference point itself is only used internally to
    anchor the search windows -- it is NOT returned, since it does not correspond
    to any meaningful phase boundary for an isometric contraction.

    :param df: DataFrame containing the EMG columns to process.
    :param fs: Sampling rate (Hz).
    :param sigma_ms: Standard deviation (ms) of the Gaussian kernel used to smooth the
        TKEO-based envelope.
    :param window_ms: Analysis window (ms), used both before the onset reference and
        after the offset reference, in which the EMG onset/offset is searched.
    :param guard_ms: Exclusion margin (ms) at the window edges to avoid spurious
        detections near the boundaries.
    :param min_window_ms: Minimum usable window length (ms) to ensure sufficient
        signal for a reliable onset/offset estimation.
    :param columns: Names of the EMG columns to process.
    :param n_reps: Expected number of isometric contractions (repetitions).
    :param threshold_k: Multiplier applied to the robust baseline spread (MAD-based) to
        set the "active" detection threshold. Increase if too many spurious segments
        are detected; decrease if real contractions are being missed.
    :param min_duration_ms: Minimum duration (ms) for a detected active segment to be
        accepted as a real contraction (filters out brief noise spikes).
    :param min_gap_ms: Maximum gap (ms) between two active segments for them to be
        merged into a single contraction (bridges short dips within a contraction).
    :param forward_margin_ms: Extra margin (ms) added past the plateau start (the
        detected threshold crossing), since the true physiological onset can occur
        slightly BEFORE that crossing.
    :param offset_margin_ms: Margin (ms) subtracted from the plateau end (the detected
        threshold crossing going down), since the true physiological offset can occur
        slightly AFTER that crossing. Defaults to the same value as forward_margin_ms
        if not provided.
    :return: List of dicts, one per (repetition, muscle), with the keys: 'rep',
        'muscle', 'onset_idx' (global index, or None), 'offset_idx' (global index, or
        None).
    """
    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in the data.")

    if offset_margin_ms is None:
        offset_margin_ms = forward_margin_ms

    window_samples = int(round((window_ms / 1000.0) * fs))
    guard_samples = int(round((guard_ms / 1000.0) * fs))
    min_window_samples = int(round((min_window_ms / 1000.0) * fs))
    forward_margin_samples = int(round((forward_margin_ms / 1000.0) * fs))
    offset_margin_samples = int(round((offset_margin_ms / 1000.0) * fs))

    envelopes = {col: _emg_envelope(df[col].to_numpy(), fs, sigma_ms) for col in columns}

    results = []
    for col in columns:
        envelope = envelopes[col]
        threshold = _robust_baseline_threshold(envelope)
        segments = _detect_active_segments(envelope, fs, threshold, min_duration_ms, min_gap_ms)

        if len(segments) != n_reps:
            print(f"Warning: expected {n_reps} contractions for '{col}', found {len(segments)}. "
                  f"Adjust threshold_k / min_duration_ms / min_gap_ms if this looks wrong. "
                  f"Keeping the {n_reps} longest segments found, in temporal order.")
            segments = sorted(
                sorted(segments, key=lambda se: se[1] - se[0], reverse=True)[:n_reps],
                key=lambda se: se[0],
            )

        for rep_num, (seg_start, seg_end) in enumerate(segments, start=1):
            # --- onset: search backward from just inside the plateau start ---
            reference_idx = min(seg_start + forward_margin_samples, envelope.size - 1)
            window_start = max(0, reference_idx - window_samples)

            envelope_window = envelope[window_start:reference_idx]
            local_onset = _find_onset_in_window(envelope_window, guard_samples, min_window_samples)

            onset_idx = None if local_onset is None else window_start + local_onset

            # --- offset: search forward from just inside the plateau end ---
            offset_anchor = max(0, seg_end - offset_margin_samples)
            offset_window_end = min(envelope.size, offset_anchor + window_samples)

            offset_window = envelope[offset_anchor:offset_window_end]
            local_offset = _find_offset_in_window(offset_window, guard_samples, min_window_samples)

            offset_idx = None if local_offset is None else offset_anchor + local_offset

            results.append({
                'rep': rep_num,
                'muscle': col,
                'onset_idx': onset_idx,
                'offset_idx': offset_idx,
            })

    return results


def detect_emg_onsets(subject_data, conditions=('HD', 'HI'), sigma_ms=30, window_ms=100, guard_ms=10,
                       min_window_ms=100, emg_columns=None, show_plot=False,
                       n_reps=3, threshold_k=4.0, min_duration_ms=300, min_gap_ms=300,
                       forward_margin_ms=0, offset_margin_ms=None, show_pipeline_plots=False,
                       pipeline_margin_before_ms=300, pipeline_margin_after_ms=200,
                       pipeline_max_reps=None, show_coarse_detection_plots=True):
    """
    Detect the EMG onset (and, for isometric conditions, offset) for both muscles,
    for every repetition, in every requested condition, using a simplified and
    deterministic version of the EstOpt algorithm.

    Two detection paths are used automatically, depending on whether the condition's
    DataFrame has a 'Phase' column:
      - If 'Phase' is present (e.g. HD), the concentric start/end of each repetition
        (shared across muscles) is used as the reference (see
        _detect_emg_onsets_single_condition). No offset detection is performed here;
        the concentric end from the Phase column is used directly.
      - If 'Phase' is absent (e.g. HI, isometric contractions), both onset and offset
        are estimated directly from the envelope, independently per muscle (see
        _detect_emg_onsets_isometric_condition). No concentric phase exists for HI,
        so no phase-boundary or onset-to-phase-start columns are produced for it.

    :param subject_data: Dict as produced/consumed by detect_concentric_eccentric_phases.
    :param conditions: Iterable with the condition names to process (e.g. 'HD', 'HI').
    :param sigma_ms: Standard deviation (ms) of the Gaussian kernel used to smooth the
        TKEO-based envelope.
    :param window_ms: Analysis window (ms) used to search for the onset/offset.
    :param guard_ms: Exclusion margin (ms) at the window edges.
    :param min_window_ms: Minimum usable window length (ms).
    :param emg_columns: Names of the EMG columns to process. Defaults to EMG_COLUMNS.
    :param show_plot: If True, plots both EMG signals with the detected onsets/offsets, per condition.
    :param n_reps: Expected number of isometric contractions per muscle (HI only).
    :param threshold_k: Robust threshold multiplier (HI only).
    :param min_duration_ms: Minimum contraction duration (ms) (HI only).
    :param min_gap_ms: Maximum gap (ms) to merge nearby active segments (HI only).
    :param forward_margin_ms: Margin (ms) added past the onset threshold crossing (HI only).
    :param offset_margin_ms: Margin (ms) subtracted from the offset threshold crossing
        (HI only). Defaults to forward_margin_ms if not provided.
    :param show_pipeline_plots: If True, additionally plots intermediate signal-processing
        steps per repetition/muscle for visual sanity-checking. For HD this shows the
        onset search (see _plot_pipeline_steps); for HI this shows both the onset and
        offset search plus the detected active segment (see _plot_hi_pipeline_steps).
    :param pipeline_margin_before_ms: Signal margin (ms) before the onset in pipeline plots.
    :param pipeline_margin_after_ms: Signal margin (ms) after the reference point in pipeline plots.
    :param pipeline_max_reps: How many repetitions to show in the pipeline plots, per muscle.
        Can be None (show all), an int (same number for every muscle), or a dict
        {muscle_column: n} to choose a different number of repetitions for each muscle.
    :param show_coarse_detection_plots: If True (HI only), plots the coarse contraction
        detection step per muscle -- raw signal, envelope + threshold + every raw
        crossing, and the final segments kept after merging/duration-filtering. Useful
        for tuning `threshold_k`, `min_duration_ms` and `min_gap_ms` before looking at
        the finer onset/offset refinement (see _plot_hi_coarse_detection).
    :return: Dict keyed by condition name, each value a pandas DataFrame with one row
        per (repetition, muscle) -- see `_results_to_dataframe`.
    """
    columns = emg_columns if emg_columns is not None else EMG_COLUMNS

    results_by_condition = {}
    for condition in conditions:
        if condition not in subject_data:
            print(f"Warning: condition '{condition}' does not exist in subject_data -- skipping.")
            continue

        df = subject_data[condition]['data']
        fs = subject_data[condition]['params']['sampling rate']

        is_isometric = 'Phase' not in df.columns

        if not is_isometric:
            condition_results = _detect_emg_onsets_single_condition(
                df, fs, sigma_ms, window_ms, guard_ms, min_window_ms, columns
            )
        else:
            condition_results = _detect_emg_onsets_isometric_condition(
                df, fs, sigma_ms, window_ms, guard_ms, min_window_ms, columns,
                n_reps, threshold_k, min_duration_ms, min_gap_ms, forward_margin_ms,
                offset_margin_ms
            )

        results_by_condition[condition] = condition_results

        if show_plot:
            _plot_onsets(df, columns, condition_results, condition)

        if is_isometric and show_coarse_detection_plots:
            _plot_hi_coarse_detection(
                df, columns, fs, sigma_ms, threshold_k, min_duration_ms, min_gap_ms, condition
            )

        if show_pipeline_plots:
            if not is_isometric:
                _plot_pipeline_steps(
                    df, columns, fs, sigma_ms, condition_results, condition,
                    margin_before_ms=pipeline_margin_before_ms,
                    margin_after_ms=pipeline_margin_after_ms,
                    max_reps=pipeline_max_reps,
                )
            else:
                _plot_hi_pipeline_steps(
                    df, columns, fs, sigma_ms, threshold_k, min_duration_ms, min_gap_ms,
                    condition_results,
                    margin_before_ms=pipeline_margin_before_ms,
                    margin_after_ms=pipeline_margin_after_ms,
                    reps_per_muscle=pipeline_max_reps,
                )

    results_dict = {condition: _results_to_dataframe(condition_results)
        for condition, condition_results in results_by_condition.items()}

    info_hd = results_dict["HD"]
    info_hi = results_dict["HI"]

    return info_hd, info_hi


def _results_to_dataframe(results):
    """
    Convert a flat list of per-(repetition, muscle) onset/offset result dicts
    (as produced by `_detect_emg_onsets_single_condition` or
    `_detect_emg_onsets_isometric_condition`) into a tidy pandas DataFrame.

    Since every dict in `results` belongs to the same condition, they all share
    the same set of keys, so the columns come out consistent automatically:
      - HD-like conditions (with a real, shared concentric phase): 'rep',
        'muscle', 'onset_idx', 'concentric_start_idx', 'concentric_end_idx',
        'onset_to_start_ms'.
      - HI-like conditions (isometric, per-muscle contraction window, no
        concentric/eccentric phase): 'rep', 'muscle', 'onset_idx', 'offset_idx'.

    :param results: List of result dicts for a single condition (possibly empty).
    :return: pandas.DataFrame with one row per (repetition, muscle), sorted by
        'rep' then 'muscle'. Empty DataFrame if `results` is empty.
    """
    df = pd.DataFrame(results)
    if df.empty:
        return df
    return df.sort_values(['rep', 'muscle']).reset_index(drop=True)


def _reps_for_muscle(muscle_results, reps_per_muscle, muscle_column):
    """
    Select how many of a muscle's repetition results to keep for plotting.

    :param muscle_results: List of result dicts already filtered to a single muscle.
    :param reps_per_muscle: None (keep all), an int (same count for every muscle), or
        a dict {muscle_column: n} (a specific count for this particular muscle).
    :param muscle_column: Name of the muscle column being plotted, used to look up
        its count when reps_per_muscle is a dict.
    :return: The (possibly truncated) list of result dicts to plot.
    """
    if reps_per_muscle is None:
        return muscle_results
    if isinstance(reps_per_muscle, dict):
        n = reps_per_muscle.get(muscle_column, len(muscle_results))
    else:
        n = reps_per_muscle
    return muscle_results[:n]


def _plot_pipeline_steps(df, columns, fs, window_ms, results, condition,
                          margin_before_ms=300, margin_after_ms=300, max_reps=None):
    """
    Plot, for every repetition and muscle, the intermediate signal-processing steps
    used to build the envelope and detect the EMG onset: the raw (filtered) EMG
    signal, the TKEO output, and the rectified TKEO (|TKEO|) overlaid with the
    smoothed envelope -- with the reference point and onset marked on every panel,
    for visual sanity-checking of the pipeline. Used for conditions with a real
    concentric phase (e.g. HD); for isometric conditions (e.g. HI) use
    `_plot_hi_pipeline_steps` instead.

    :param df: DataFrame containing the EMG columns to plot.
    :param columns: List of EMG column names to plot.
    :param fs: Sampling rate (Hz).
    :param window_ms: Moving-average window (ms) used to build the envelope (must
        match the value used for detection, i.e. sigma_ms).
    :param results: List of onset result dicts (must contain 'concentric_start_idx'
        and 'concentric_end_idx').
    :param condition: Name of the condition being plotted (used in the plot title).
    :param margin_before_ms: How much signal (ms) to show before the onset.
    :param margin_after_ms: How much signal (ms) to show after the reference point.
    :param max_reps: How many repetitions to plot, per muscle. None (show all), an
        int (same count for every muscle), or a dict {muscle_column: n}.
    :return: None. Displays one figure per (muscle, repetition).
    """
    margin_before_samples = int(round((margin_before_ms / 1000.0) * fs))
    margin_after_samples = int(round((margin_after_ms / 1000.0) * fs))

    for col in columns:
        raw = df[col].to_numpy()
        rectified = np.abs(raw)
        tkeo = _tkeo(rectified)
        tkeo_rectified = np.abs(tkeo)

        window_samples = max(1, int(round((window_ms / 1000.0) * fs)))
        kernel = np.ones(window_samples) / window_samples
        envelope = np.convolve(tkeo_rectified, kernel, mode="same")

        muscle_results = [r for r in results if r['muscle'] == col]
        muscle_results = _reps_for_muscle(muscle_results, max_reps, col)

        for r in muscle_results:
            ref_idx = r['concentric_start_idx']
            onset_idx = r['onset_idx']
            offset_idx = r.get('offset_idx')

            if offset_idx is None:
                offset_idx = r.get("concentric_end_idx")

            if onset_idx is None or offset_idx is None:
                continue

            plot_start = max(0, onset_idx - margin_before_samples)
            plot_end = min(raw.size, offset_idx + margin_after_samples)
            sl = slice(plot_start, plot_end)
            t = np.arange(plot_start, plot_end)

            fig, axes = plt.subplots(4, 1, figsize=(10, 7.5), sharex=True)

            axes[0].plot(t, raw[sl], color='black', linewidth=0.8)
            axes[0].set_ylabel('Raw EMG\n(filtered)', fontsize=8)

            axes[1].plot(t, rectified[sl], color='tab:purple', linewidth=0.8)
            axes[1].set_ylabel('Rectified EMG', fontsize=8)

            axes[2].plot(t, tkeo[sl], color='black', linewidth=0.6, alpha=0.6, label='TKEO')
            axes[2].set_ylabel('TKEO', fontsize=8)

            axes[3].plot(t, tkeo_rectified[sl], color='tab:gray', linewidth=1.6, label='|TKEO|')
            axes[3].set_ylabel('|TKEO|', fontsize=8)
            axes[3].plot(t, envelope[sl], color='tab:orange', linewidth=1.6, label='envelope')

            for ax in axes:
                ax.axvline(ref_idx, color='tab:blue', linestyle='--', alpha=0.6)
                ax.axvline(onset_idx, color='tab:red', linestyle='-', alpha=0.8)
                if offset_idx is not None:
                    ax.axvline(offset_idx, color='tab:green', linestyle='-', alpha=0.8)

            axes[-1].set_xlabel('Samples')
            fig.suptitle(
                f"{condition} -- {col} -- Rep {r['rep']} -- pipeline steps\n"
                f"(blue dashed = reference point, red = onset, green = end of concentric phase)",
                fontsize=10,
            )
            plt.tight_layout()
            plt.show()


def _plot_hi_pipeline_steps(df, columns, fs, sigma_ms, threshold_k, min_duration_ms, min_gap_ms,
                             results, margin_before_ms=300, margin_after_ms=300,
                             reps_per_muscle=None):
    """
    Plot, for every repetition and muscle, the intermediate signal-processing steps
    used to detect the EMG onset AND offset in isometric conditions (e.g. HI): the
    raw (filtered) EMG signal, the rectified signal, the TKEO output, and the
    rectified TKEO (|TKEO|) overlaid with the smoothed envelope and the robust
    detection threshold -- with the detected active ("plateau") segment shaded and
    the refined onset/offset marked on every panel, for visual sanity-checking of
    the full HI pipeline (threshold-based segment detection -> onset/offset refinement).

    :param df: DataFrame containing the EMG columns to plot.
    :param columns: List of EMG column names to plot.
    :param fs: Sampling rate (Hz).
    :param sigma_ms: Moving-average window (ms) used to build the envelope (must
        match the value used for detection).
    :param threshold_k: Multiplier applied to the robust baseline spread (MAD-based)
        to set the "active" detection threshold (must match the value used for
        detection, so the plotted threshold/segments match the actual onset_idx/
        offset_idx values).
    :param min_duration_ms: Minimum duration (ms) for a detected active segment to be
        accepted as a real contraction (must match the value used for detection).
    :param min_gap_ms: Maximum gap (ms) between two active segments for them to be
        merged into a single contraction (must match the value used for detection).
    :param results: List of onset/offset result dicts for the HI condition (must
        contain 'rep', 'muscle', 'onset_idx', 'offset_idx').
    :param margin_before_ms: How much signal (ms) to show before the onset.
    :param margin_after_ms: How much signal (ms) to show after the offset.
    :param reps_per_muscle: How many repetitions to plot, per muscle. None (show
        all), an int (same count for every muscle), or a dict {muscle_column: n} to
        choose a different number of repetitions for each muscle.
    :return: None. Displays one figure per (muscle, repetition).
    """
    margin_before_samples = int(round((margin_before_ms / 1000.0) * fs))
    margin_after_samples = int(round((margin_after_ms / 1000.0) * fs))

    for col in columns:
        raw = df[col].to_numpy()
        rectified = np.abs(raw)
        tkeo = _tkeo(rectified)
        tkeo_rectified = np.abs(tkeo)
        envelope = _emg_envelope(raw, fs, sigma_ms)

        threshold = _robust_baseline_threshold(envelope)
        segments = _detect_active_segments(envelope, fs, threshold, min_duration_ms, min_gap_ms)

        muscle_results = [r for r in results if r['muscle'] == col]
        muscle_results = _reps_for_muscle(muscle_results, reps_per_muscle, col)

        for r in muscle_results:
            onset_idx = r['onset_idx']
            offset_idx = r['offset_idx']

            if onset_idx is None or offset_idx is None:
                continue

            plot_start = max(0, onset_idx - margin_before_samples)
            plot_end = min(raw.size, offset_idx + margin_after_samples)
            sl = slice(plot_start, plot_end)
            t = np.arange(plot_start, plot_end)

            fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

            axes[0].plot(t, raw[sl], color='black', linewidth=0.8)
            axes[0].set_ylabel('Raw EMG\n(filtered)', fontsize=8)

            axes[1].plot(t, rectified[sl], color='tab:purple', linewidth=0.8)
            axes[1].set_ylabel('Rectified EMG', fontsize=8)

            axes[2].plot(t, tkeo[sl], color='black', linewidth=0.6, alpha=0.6)
            axes[2].set_ylabel('TKEO', fontsize=8)

            axes[3].plot(t, tkeo_rectified[sl], color='tab:gray', linewidth=1.2, label='|TKEO|')
            axes[3].plot(t, envelope[sl], color='tab:orange', linewidth=1.6, label='envelope')
            axes[3].axhline(threshold, color='tab:brown', linestyle=':', linewidth=1.2, label='threshold')
            axes[3].set_ylabel('|TKEO| / envelope', fontsize=8)
            axes[3].legend(fontsize=7, loc='upper right')

            # shade the detected active ("plateau") segment(s) that overlap the plotted window
            for seg_start, seg_end in segments:
                if seg_end < plot_start or seg_start > plot_end:
                    continue
                for ax in axes:
                    ax.axvspan(max(seg_start, plot_start), min(seg_end, plot_end),
                               color='tab:blue', alpha=0.08)

            for ax in axes:
                ax.axvline(onset_idx, color='tab:red', linestyle='-', alpha=0.8)
                ax.axvline(offset_idx, color='tab:green', linestyle='-', alpha=0.8)

            axes[-1].set_xlabel('Samples')
            fig.suptitle(
                f"HI -- {col} -- Rep {r['rep']} -- pipeline steps\n"
                f"(shaded = detected active segment, dotted = threshold, "
                f"red = onset, green = offset)",
                fontsize=10,
            )
            plt.tight_layout()
            plt.show()


def _plot_onsets(df, columns, results, condition):
    """
    Plot each EMG channel with the detected onsets, offsets (when available) and
    reference-point markers (when available -- e.g. not for isometric conditions).

    :param df: DataFrame containing the EMG columns to plot.
    :param columns: List of EMG column names to plot.
    :param results: List of onset/offset result dicts.
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

        col_results = [r for r in results if r['muscle'] == col]
        for i, r in enumerate(col_results):
            show_label = i == 0
            reference_idx = r.get('concentric_start_idx')
            if reference_idx is not None:
                ax.axvline(reference_idx, color='tab:blue', linestyle='--', alpha=0.6,
                           label='Reference point' if show_label else None)
            if r['onset_idx'] is not None:
                ax.plot(r['onset_idx'], signal[r['onset_idx']], 'r^', markersize=8,
                       label='EMG onset' if show_label else None)
            if r.get('offset_idx') is not None:
                ax.plot(r['offset_idx'], signal[r['offset_idx']], 'gv', markersize=8,
                       label='EMG offset' if show_label else None)

        ax.set_ylabel(col, fontsize=8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=8)
    axes[-1].set_xlabel('Samples')
    fig.suptitle(f'EMG Onset/Offset Detection -- {condition} Condition (all repetitions)')
    plt.tight_layout()
    plt.show()