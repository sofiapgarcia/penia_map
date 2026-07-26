import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from constants import FORCE_COLUMN, DISPLACEMENT_COLUMN , EMG_Extensor_filt , EMG_Flexor_filt

FORCE_DERIVATIVE = "Force_derivative"

DEFAULT_PLOT_COLUMNS = [EMG_Extensor_filt, EMG_Flexor_filt, DISPLACEMENT_COLUMN, FORCE_COLUMN, FORCE_DERIVATIVE]

def detect_concentric_eccentric_phases(subject_data, show_plot=True,
                                        derivative_sigma_ms=20, n_reps=12,
                                        min_distance_s=1.0, threshold_fraction=0.2):
    """
    Detects the concentric and eccentric phases for EACH repetition, using the
    positive peaks of the force derivative (HD condition only).

    Each repetition is identified from a positive peak of the force
    derivative: the n_reps most prominent peaks are selected (corresponding to
    the n_reps repetitions of the protocol). For each repetition, the movement
    onset and offset are found using a threshold defined as a fixed fraction
    of THAT repetition's own peak derivative -- so the detection adapts
    locally to the intensity of each individual movement, rather than using a
    single threshold for the whole signal or across participants.

    For each peak:
      - search BACKWARD until the derivative returns close to zero (below the
        local threshold) -> onset (start of the repetition / concentric phase).
      - search FORWARD until the derivative changes sign -> turn (end of
        concentric / start of eccentric phase).
      - keep searching FORWARD from the turn -- past the eccentric trough --
        until the derivative returns close to zero again -> offset (end of
        the repetition).

    :param subject_data: dict with condition-keyed data. subject_data['HD']['data'] must be a
        DataFrame and subject_data['HD']['params']['sampling_rate'] the sampling frequency.
    :param show_plot: bool, if True plots the selected channels with every phase shaded
        and marked.
    :param derivative_sigma_ms: Standard deviation (ms) of the Gaussian kernel used to
        smooth the force derivative signal before peak detection.
    :param n_reps: Expected number of repetitions (peaks) to detect.
    :param min_distance_s: Minimum time (s) required between two consecutive peaks,
        to avoid detecting spurious peaks within the same repetition. Should be
        shorter than the true interval between repetitions.
    :param threshold_fraction: Fraction of each repetition's own peak derivative
        used as that repetition's local onset/offset threshold (e.g. 0.1 = 10% of
        the peak). Increase if onsets/offsets are detected too early/late due to
        noise; decrease if the true onset/offset is being missed.
    :return: subject_data, with an added 'Phase' column in subject_data['HD']['data']
        marking each repetition's concentric ('C1', 'C2', ...) and eccentric ('E1', 'E2', ...)
        samples, everything else marked as 'R'. A 'Force_derivative' column is also
        added for inspection/debugging purposes.
    """
    if 'HD' not in subject_data:
        raise KeyError("Condition 'HD' does not exist in subject_data.")

    df = subject_data['HD']['data']
    fs = subject_data['HD']['params']['sampling rate']
    force = df[FORCE_COLUMN].to_numpy()

    derivative = _compute_force_derivative(force, fs, derivative_sigma_ms)

    peak_indices = _find_rep_peaks(derivative, fs, n_reps, min_distance_s)

    if peak_indices.size == 0:
        raise ValueError("No repetition peaks were detected in the force derivative.")
    if peak_indices.size < n_reps:
        print(f"Warning: expected {n_reps} repetitions, only found {peak_indices.size} "
              f"peaks. Adjust min_distance_s / derivative_sigma_ms if this looks wrong.")

    phases_list = []
    n_peaks = peak_indices.size
    for rep_i, peak_idx in enumerate(peak_indices):
        rep_num = rep_i + 1
        peak_value = derivative[peak_idx]
        threshold = threshold_fraction * peak_value

        # bound each repetition's search window by its neighboring peaks,
        # so it never bleeds into the previous/next repetition
        search_start = 0 if rep_i == 0 else peak_indices[rep_i - 1]
        search_end = (derivative.size - 1) if rep_i == n_peaks - 1 else peak_indices[rep_i + 1]

        onset = _find_onset(derivative, peak_idx, threshold, search_start)
        turn = _find_turn(derivative, peak_idx, search_end)
        offset = _find_offset(derivative, turn, threshold, search_end)

        phases_list.append({
            'rep': rep_num,
            'concentric_start': int(onset),
            'concentric_end': int(turn),
            'eccentric_start': int(turn),
            'eccentric_end': int(offset),
            'peak_idx': int(peak_idx),
            'peak_derivative': float(peak_value),
            'derivative_threshold': float(threshold),
            'peak_value': float(force[turn]),
            'fs': fs,
        })

    df['Force_derivative'] = derivative
    df['Phase'] = 'R'
    for phase in phases_list:
        rep_num = phase['rep']
        df.iloc[phase['concentric_start']:phase['concentric_end'], df.columns.get_loc('Phase')] = f'C{rep_num}'
        df.iloc[phase['eccentric_start']:phase['eccentric_end'] + 1, df.columns.get_loc('Phase')] = f'E{rep_num}'

    subject_data['HD']['data'] = df

    if show_plot:
        _plot_phases(df, phases_list)

    return subject_data

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def _compute_force_derivative(force, fs, sigma_ms, plot=True):
    """
    Compute a smoothed rate-of-change (derivative) signal from the force
    signal via numerical differentiation, followed by light Gaussian
    smoothing to suppress sample-to-sample noise (which would otherwise
    cause spurious peak detections).

    :param force: 1D array with the force signal.
    :param fs: Sampling rate (Hz).
    :param sigma_ms: Standard deviation (ms) of the Gaussian smoothing kernel.
    :param plot: If True, plot the force signal and its derivative.
    :return: 1D array with the smoothed force derivative (same length as force).
    """
    derivative = np.gradient(force) * fs

    sigma_samples = (sigma_ms / 1000.0) * fs
    derivative_smooth = gaussian_filter1d(derivative, sigma=sigma_samples)

    return derivative_smooth


def _find_rep_peaks(derivative, fs, n_reps, min_distance_s):
    """
    Find the n_reps most prominent positive peaks in the force derivative,
    each corresponding approximately to the onset of a repetition's concentric
    phase. If more than n_reps peaks satisfy the minimum-distance constraint,
    only the n_reps peaks with the largest prominence are kept.

    :param derivative: 1D array with the force derivative signal.
    :param fs: Sampling rate (Hz).
    :param n_reps: Expected number of repetitions (peaks) to keep.
    :param min_distance_s: Minimum distance (s) required between neighboring peaks.
    :return: Sorted (temporal order) 1D array of peak sample indices, length <= n_reps.
    """
    min_distance_samples = max(1, int(round(min_distance_s * fs)))

    peak_indices, properties = find_peaks(derivative, distance=min_distance_samples,
                                           prominence=0)

    if peak_indices.size == 0:
        return peak_indices

    if peak_indices.size > n_reps:
        top_n = np.argsort(properties['prominences'])[-n_reps:]
        peak_indices = peak_indices[top_n]

    return np.sort(peak_indices)


def _find_onset(derivative, peak_idx, threshold, search_start):
    """
    Search BACKWARD from a repetition's peak derivative until the derivative
    drops back to (or below) the local threshold, marking the movement onset
    (start of the repetition).

    :param derivative: 1D array with the force derivative signal.
    :param peak_idx: Sample index of this repetition's peak derivative.
    :param threshold: Local threshold (fraction of this peak's amplitude).
    :param search_start: Earliest sample index the search is allowed to reach
        (e.g. the previous repetition's offset, or 0).
    :return: Sample index of the onset, or `search_start` if the derivative
        never drops below threshold within the allowed search range.
    """
    idx = peak_idx
    while idx > search_start and derivative[idx] > threshold:
        idx -= 1
    return idx


def _find_turn(derivative, peak_idx, search_end):
    """
    Search FORWARD from a repetition's peak derivative until the derivative
    changes sign (crosses zero), marking the end of the concentric phase /
    start of the eccentric phase.

    :param derivative: 1D array with the force derivative signal.
    :param peak_idx: Sample index of this repetition's peak derivative.
    :param search_end: Latest sample index the search is allowed to reach.
    :return: Sample index of the turn, or `search_end` if the derivative never
        turns negative within the allowed search range.
    """
    idx = peak_idx
    while idx < search_end and derivative[idx] > 0:
        idx += 1
    return idx


def _find_offset(derivative, turn_idx, threshold, search_end):
    """
    Search FORWARD from the concentric-to-eccentric turn until the derivative
    returns to (near) zero A SECOND TIME -- i.e. after dipping below the
    (negative) local threshold during the eccentric deceleration, and then
    rising back above it.

    This two-stage search is needed because right at `turn_idx` the derivative
    has *just* crossed zero and is therefore already close to it: a naive
    single-stage "stop once |derivative| <= threshold" check would trigger
    immediately at the turn itself, without ever letting the signal actually
    dip into the eccentric trough. So the search first pushes forward until
    the derivative genuinely drops below -threshold (confirming we've left the
    near-zero band), and only then looks for the point where it climbs back
    above -threshold again -- that second crossing is the true offset.

    :param derivative: 1D array with the force derivative signal.
    :param turn_idx: Sample index of the concentric-to-eccentric turn.
    :param threshold: Local threshold (fraction of this repetition's peak amplitude).
    :param search_end: Latest sample index the search is allowed to reach
        (e.g. the next repetition's onset, or the end of the signal).
    :return: Sample index of the offset, or `search_end` if the derivative
        never completes both stages within the allowed search range.
    """
    idx = turn_idx

    # stage 1: push forward until the derivative genuinely dips below -threshold
    while idx < search_end and derivative[idx] > -threshold:
        idx += 1

    # stage 2: continue forward until it climbs back above -threshold (near zero again)
    while idx < search_end and derivative[idx] <= -threshold:
        idx += 1

    return idx

def _plot_phases(df, phases_list, plot_columns=DEFAULT_PLOT_COLUMNS):
    """
    Plots the selected channels (EMG / force / displacement / force derivative
    by default) with the concentric and eccentric phases of every detected
    repetition shaded, marking the start/end points of each phase as vertical
    lines. For the 'Force_derivative' channel specifically, also marks each
    repetition's peak and its local +/- threshold.
    """
    n = len(plot_columns)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.8 * n), sharex=True)
    if n == 1:
        axes = [axes]

    labels = {
        'concentric': 'Concentric phase',
        'eccentric': 'Eccentric phase',
        'start': 'Onset',
        'turn': 'Direction change',
        'end': 'Offset',
        'peak': 'Peak derivative',
    }

    for ax, ch in zip(axes, plot_columns):
        signal = df[ch].to_numpy()
        ax.plot(signal, color='black', linewidth=0.8)

        for rep_i, phase in enumerate(phases_list):
            cs, ce = phase['concentric_start'], phase['concentric_end']
            es, ee = phase['eccentric_start'], phase['eccentric_end']
            show_label = rep_i == 0

            ax.axvspan(cs, ce, color='tab:blue', alpha=0.2, label=labels['concentric'] if show_label else None)
            ax.axvspan(es, ee, color='tab:orange', alpha=0.2, label=labels['eccentric'] if show_label else None)
            ax.axvline(cs, color='tab:green', linestyle='-', alpha=0.8, label=labels['start'] if show_label else None)
            ax.axvline(ce, color='tab:red', linestyle='-', alpha=0.8, label=labels['turn'] if show_label else None)
            ax.axvline(ee, color='tab:purple', linestyle='-', alpha=0.8, label=labels['end'] if show_label else None)

            if ch == 'Force_derivative':
                threshold = phase['derivative_threshold']
                ax.hlines(threshold, cs, ee, color='tab:green', linestyle=':', linewidth=1,
                          label='local threshold' if show_label else None)
                ax.hlines(-threshold, cs, ee, color='tab:green', linestyle=':', linewidth=1)

        ax.set_ylabel(ch, fontsize=8)

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc='upper right', fontsize=8)
    axes[-1].set_xlabel('Samples')
    fig.suptitle('Concentric and Eccentric Phases - HD Condition')
    plt.tight_layout()
    plt.show()