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
    tkeo = _tkeo(x)
    rectified = np.abs(tkeo)
    sigma_samples = (sigma_ms / 1000.0) * fs
    return gaussian_filter1d(rectified, sigma=sigma_samples)


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


def _robust_baseline_threshold(envelope, threshold_k):
    """
    Estimate a robust detection threshold for an envelope signal, based on the median
    and the median absolute deviation (MAD).

    :param envelope: 1D array with the (smoothed) energy envelope.
    :param threshold_k: Multiplier applied to the robust spread estimate.
    :return: Scalar threshold value.
    """
    baseline = np.median(envelope)
    mad = np.median(np.abs(envelope - baseline))
    robust_std = mad * 1.4826
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


def _detect_emg_onsets_isometric_condition(df, fs, sigma_ms, window_ms, guard_ms, min_window_ms,
                                            columns, n_reps, threshold_k, min_duration_ms,
                                            min_gap_ms, forward_margin_ms, offset_margin_ms=None):
    """
    Detect EMG onsets AND offsets for isometric conditions (e.g. HI), where there is
    no 'Phase' column to mark a concentric start/end, since there's no concentric
    phase at all.

    Instead, a reference point per repetition is estimated directly from the
    envelope: each of the `n_reps` isometric contractions shows up as a period where
    the envelope stays above a robust, data-driven threshold (a "plateau").
      - The onset is refined by searching BACKWARD from the plateau start for the
        sample that best splits a constant baseline segment from a rising segment
        (same as the phase-based detection).
      - The offset is refined by searching FORWARD from the plateau end for the
        sample that best splits a falling (deactivation) segment from a constant
        post-contraction baseline segment -- the mirror image of the onset search.

    Because the envelope is computed independently per muscle, the plateau (and
    therefore both onset and offset) can fall at slightly different sample indices
    for each muscle.

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
        None), 'concentric_start_idx' (plateau-start reference point, kept for
        backward compatibility -- NOT a true concentric phase), 'onset_to_start_ms'.
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

        for rep_num, (seg_start, seg_end) in enumerate(segments, start=1):
            # --- onset: search backward from just inside the plateau start ---
            reference_idx = min(seg_start + forward_margin_samples, envelope.size - 1)
            window_start = max(0, reference_idx - window_samples)

            envelope_window = envelope[window_start:reference_idx]
            local_onset = _find_onset_in_window(envelope_window, guard_samples, min_window_samples)

            if local_onset is None:
                onset_idx = None
                onset_to_start_ms = None
            else:
                onset_idx = window_start + local_onset
                onset_to_start_ms = (reference_idx - onset_idx) / fs * 1000.0

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
                'concentric_start_idx': reference_idx,
                'onset_to_start_ms': onset_to_start_ms,
            })

    return results


def detect_emg_onsets(subject_data, conditions=('HD', 'HI'), sigma_ms=50, window_ms=300, guard_ms=10,
                       min_window_ms=100, emg_columns=None, show_plot=True,
                       n_reps=3, threshold_k=4.0, min_duration_ms=300, min_gap_ms=300,
                       forward_margin_ms=50, offset_margin_ms=None, show_pipeline_plots=False,
                       pipeline_margin_before_ms=300, pipeline_margin_after_ms=200,
                       pipeline_max_reps=None):
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
        _detect_emg_onsets_isometric_condition).

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
        steps per repetition/muscle for visual sanity-checking.
    :param pipeline_margin_before_ms: Signal margin (ms) before the onset in pipeline plots.
    :param pipeline_margin_after_ms: Signal margin (ms) after the reference point in pipeline plots.
    :param pipeline_max_reps: If set, limits repetitions plotted per muscle in pipeline plots.
    :return: Dict keyed by condition name, each value a list of per-(rep, muscle) result dicts
        (see _detect_emg_onsets_single_condition / _detect_emg_onsets_isometric_condition).
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
                n_reps, threshold_k, min_duration_ms, min_gap_ms, forward_margin_ms,
                offset_margin_ms
            )

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

    return reshape_onset_results(results_by_condition)


def reshape_onset_results(results_by_condition):
    """
    Reestrutura o output de detect_emg_onsets (uma lista plana de dicts por
    condicao, um por (rep, muscle)) num dicionario aninhado organizado por
    condicao -> repeticao -> {'onset': {...}, <fase>: {...}}.

    Duas condicoes sao tratadas de forma diferente:
      - Condicoes com fase concentrica real e partilhada entre musculos (ex: HD,
        identificada pela presenca da chave 'concentric_end_idx' nos resultados):
        a chave e 'concentric_phase', um unico dict {'start': ..., 'end': ...}
        por repeticao (e mesmo uma fase concentrica).
      - Condicoes isometricas sem fase concentrica real (ex: HI, identificada pela
        ausencia de 'concentric_end_idx' mas presenca de 'offset_idx'): a chave e
        'contraction' (nao 'concentric_phase', porque nao ha fase concentrica),
        organizada por musculo, com 'contraction_beginning' / 'contraction_end'
        em vez de 'start' / 'end', ja que o inicio e o fim sao calculados de
        forma independente por musculo.

    :param results_by_condition: Dict tal como devolvido por detect_emg_onsets.
    :return: Dict no formato:
        {
            condicao: {
                rep: {
                    'onset': {musculo: onset_idx, ...},
                    'concentric_phase': {'start': idx, 'end': idx}   # ex: HD
                        OU
                    'contraction': {                                  # ex: HI
                        musculo: {
                            'contraction_beginning': idx,
                            'contraction_end': idx,
                        },
                        ...
                    }
                },
                ...
            },
            ...
        }
    """
    reshaped = {}

    for condition, results in results_by_condition.items():
        condition_dict = {}

        # Deteta o "tipo" de condicao a partir da primeira entrada de resultados.
        has_shared_phase = bool(results) and 'concentric_end_idx' in results[0]

        for r in results:
            rep = r['rep']
            muscle = r['muscle']

            if rep not in condition_dict:
                if has_shared_phase:
                    condition_dict[rep] = {
                        'onset': {},
                        'concentric_phase': {
                            'start': r.get('concentric_start_idx'),
                            'end': r.get('concentric_end_idx'),
                        },
                    }
                else:
                    condition_dict[rep] = {
                        'onset': {},
                        'contraction': {},
                    }

            condition_dict[rep]['onset'][muscle] = r['onset_idx']

            if not has_shared_phase:
                # HI-like: inicio/fim da contracao calculados por musculo.
                condition_dict[rep]['contraction'][muscle] = {
                    'contraction_beginning': r.get('concentric_start_idx'),
                    'contraction_end': r.get('offset_idx'),
                }

        reshaped[condition] = condition_dict

    return reshaped


def _plot_pipeline_steps(df, columns, fs, sigma_ms, results, condition,
                          margin_before_ms=300, margin_after_ms=200, max_reps=None):
    """
    Plot, for every repetition and muscle, the intermediate signal-processing steps
    used to build the envelope and detect the EMG onset/offset: the raw (filtered)
    EMG signal, the TKEO output, and the rectified TKEO (|TKEO|) overlaid with the
    smoothed envelope -- with the reference point, onset and (when available) offset
    marked on every panel, for visual sanity-checking of the pipeline.

    :param df: DataFrame containing the EMG columns to plot.
    :param columns: List of EMG column names to plot.
    :param fs: Sampling rate (Hz).
    :param sigma_ms: Standard deviation (ms) of the Gaussian kernel used to smooth the
        TKEO-based envelope (must match the value used for detection).
    :param results: List of onset/offset result dicts.
    :param condition: Name of the condition being plotted (used in the plot title).
    :param margin_before_ms: How much signal (ms) to show before the onset.
    :param margin_after_ms: How much signal (ms) to show after the reference point
        (or after the offset, when available).
    :param max_reps: If set, limits the number of repetitions plotted per muscle.
    :return: None. Displays one figure per (muscle, repetition).
    """
    margin_before_samples = int(round((margin_before_ms / 1000.0) * fs))
    margin_after_samples = int(round((margin_after_ms / 1000.0) * fs))

    for col in columns:
        raw = df[col].to_numpy()

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
            offset_idx = r.get('offset_idx')

            anchor = onset_idx if onset_idx is not None else ref_idx
            end_anchor = offset_idx if offset_idx is not None else ref_idx
            plot_start = max(0, anchor - margin_before_samples)
            plot_end = min(raw.size, end_anchor + margin_after_samples)
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
                if offset_idx is not None:
                    ax.axvline(offset_idx, color='tab:green', linestyle='-', alpha=0.8)

            axes[-1].set_xlabel('Samples')
            fig.suptitle(
                f"{condition} -- {col} -- Rep {r['rep']} -- pipeline steps\n"
                f"(blue dashed = reference point, red = onset, green = offset)",
                fontsize=10,
            )
            plt.tight_layout()
            plt.show()


def _plot_onsets(df, columns, results, condition):
    """
    Plot each EMG channel with the detected onsets, offsets (when available) and
    reference-point markers.

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
            ax.axvline(r['concentric_start_idx'], color='tab:blue', linestyle='--', alpha=0.6,
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