import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from constants import FORCE_COLUMN, DISPLACEMENT_COLUMN

DEFAULT_PLOT_COLUMNS = ['EMG_Extensor_filt', 'EMG_Flexor_filt', DISPLACEMENT_COLUMN, FORCE_COLUMN]


def _estimate_baseline(force):
    """Most frequent value of the signal (rounded to absorb noise)."""
    rounded = np.round(force, decimals=1)
    mode_result = stats.mode(rounded, keepdims=False)
    return float(np.ravel(mode_result.mode)[0])


def _find_reps(force, threshold):
    """
    Finds where the signal crosses the threshold going up (rep start) and
    going down (rep end), and pairs them up.

    :param force: 1D array with the force signal.
    :param threshold: value above which the signal is considered "in a rep".
    :return: list of (start, end) index pairs, one per repetition.
    """
    above = force > threshold
    edges = np.diff(above.astype(int))

    starts = np.flatnonzero(edges == 1) + 1   # crosses threshold going up
    ends = np.flatnonzero(edges == -1) + 1    # crosses threshold going down

    # drop a leading "end" that isn't preceded by a "start"
    if ends.size and starts.size and ends[0] < starts[0]:
        ends = ends[1:]
    # drop a trailing "start" that never comes back down
    starts = starts[:ends.size]

    return list(zip(starts, ends))


def _find_turn(force, start, end):
    """
    First index in [start, end] where the signal stops rising, i.e. the first
    point after which force decreases -- even if only by a single sample.
    Marks the end of the concentric phase / start of the eccentric phase.

    If the signal rises the whole way without ever turning, returns `end`.
    """
    diffs = np.diff(force[start:end + 1])
    falling = diffs < 0
    if falling.any():
        return start + int(np.argmax(falling))
    return end


def _select_plot_channels(df, plot_columns=None):
    """
    Picks which columns to plot. If plot_columns is given explicitly, use those;
    otherwise use DEFAULT_PLOT_COLUMNS (EMG extensor, EMG flexor, displacement,
    force), keeping only the ones that actually exist in df.
    """
    columns = plot_columns if plot_columns is not None else DEFAULT_PLOT_COLUMNS
    return [c for c in columns if c in df.columns]


def detect_concentric_eccentric_phases(subject_data, show_plot=True, plot_columns=None):
    """
    Detects the concentric and eccentric phases for EACH repetition present in the force
    signal, using only the HD condition.

    Repetitions are segmented from threshold crossings around the baseline: a rep starts
    when the force rises above the threshold and ends when it falls back below it.

    Concentric phase: from the start of the rep until the first direction change (the
    first instant the force stops rising, however small that dip is).
    Eccentric phase: from that turning point until the force falls back below threshold.

    :param subject_data: dict with condition-keyed data. subject_data['HD']['data'] must be a
        DataFrame and subject_data['HD']['params']['sampling_rate'] the sampling frequency.
    :param show_plot: bool, if True plots the selected channels with every phase shaded
        and marked.
    :param plot_columns: optional explicit list of column names to plot. If None,
        defaults to DEFAULT_PLOT_COLUMNS (EMG extensor, EMG flexor, displacement,
        force).
    :return: subject_data, with an added 'Phase' column in subject_data['HD']['data']
        marking each repetition's concentric ('C1', 'C2', ...) and eccentric ('E1', 'E2', ...)
        samples, everything else marked as 'R'.
    """
    if 'HD' not in subject_data:
        raise KeyError("Condition 'HD' does not exist in subject_data.")

    df = subject_data['HD']['data']
    fs = subject_data['HD']['params']['sampling rate']
    force = df[FORCE_COLUMN].to_numpy()

    baseline = _estimate_baseline(force)
    threshold = baseline + 0.05 * (force.max() - baseline)

    reps = _find_reps(force, threshold)
    if not reps:
        raise ValueError("No repetition was detected above the defined threshold.")

    phases_list = []
    for rep_num, (start, end) in enumerate(reps, start=1):
        turn = _find_turn(force, start, end)
        phases_list.append({
            'rep': rep_num,
            'concentric_start': int(start),
            'concentric_end': int(turn),
            'eccentric_start': int(turn),
            'eccentric_end': int(end),
            'baseline': baseline,
            'peak_value': float(force[turn]),
            'fs': fs,
        })

    df['Phase'] = 'R'
    for phase in phases_list:
        rep_num = phase['rep']
        df.iloc[phase['concentric_start']:phase['concentric_end'], df.columns.get_loc('Phase')] = f'C{rep_num}'
        df.iloc[phase['eccentric_start']:phase['eccentric_end'] + 1, df.columns.get_loc('Phase')] = f'E{rep_num}'

    subject_data['HD']['data'] = df

    if show_plot:
        _plot_phases(df, phases_list, plot_columns)

    return subject_data


def _plot_phases(df, phases_list, plot_columns=None):
    """
    Plots the selected channels (EMG / force / displacement by default) with the
    concentric and eccentric phases of every detected repetition shaded, marking the
    start/end points of each phase.
    """
    channels = _select_plot_channels(df, plot_columns)
    n = len(channels)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.8 * n), sharex=True)
    if n == 1:
        axes = [axes]

    labels = {
        'concentric': 'Concentric phase',
        'eccentric': 'Eccentric phase',
        'start': 'Concentric start',
        'turn': 'Direction change',
        'end': 'Eccentric end',
    }

    for ax, ch in zip(axes, channels):
        signal = df[ch].to_numpy()
        ax.plot(signal, color='black', linewidth=0.8)

        for rep_i, phase in enumerate(phases_list):
            cs, ce = phase['concentric_start'], phase['concentric_end']
            es, ee = phase['eccentric_start'], phase['eccentric_end']
            show_label = rep_i == 0

            ax.axvspan(cs, ce, color='tab:blue', alpha=0.2, label=labels['concentric'] if show_label else None)
            ax.axvspan(es, ee, color='tab:orange', alpha=0.2, label=labels['eccentric'] if show_label else None)
            ax.plot(cs, signal[cs], 'go', markersize=6, label=labels['start'] if show_label else None)
            ax.plot(ce, signal[ce], 'ro', markersize=6, label=labels['turn'] if show_label else None)
            ax.plot(ee, signal[ee], 'ms', markersize=6, label=labels['end'] if show_label else None)

        ax.set_ylabel(ch, fontsize=8)

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc='upper right', fontsize=8)
    axes[-1].set_xlabel('Samples')
    fig.suptitle('Concentric and Eccentric Phases -- HD Condition (all repetitions)')
    plt.tight_layout()
    plt.show()