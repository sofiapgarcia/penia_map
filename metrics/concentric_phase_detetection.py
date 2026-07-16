import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import find_peaks

from constants import FORCE_COLUMN


def detect_concentric_eccentric_phases(subject_data, min_rep_distance_s=1.0,
                                        min_prominence_ratio=0.3, show_plot=True):
    """
    Detects the concentric and eccentric phases for EACH repetition present in the force
    signal, using only the HD condition.

    Concentric phase: from the moment the force starts rising above baseline until it reaches
    the local peak value of that repetition.
    Eccentric phase: from that peak value until the force returns to baseline again.

    :param subject_data: dict with condition-keyed data. subject_data['HD']['data'] must be a
        DataFrame and subject_data['HD']['params']['sampling_rate'] the sampling frequency.
    :param min_rep_distance_s: minimum time (seconds) expected between two consecutive
        repetition peaks. Used to avoid detecting noise as separate reps.
    :param min_prominence_ratio: minimum peak prominence, as a fraction of (global_peak -
        baseline), required for a local maximum to be considered a repetition.
    :param show_plot: bool, if True plots all channels with every phase shaded and marked.
    :return: list of dicts (one per detected repetition), each with 'concentric_start',
        'concentric_end' (=peak), 'eccentric_start' (=peak), 'eccentric_end' (sample indices),
        plus 'baseline', 'peak_value' and 'fs'.
    """
    if 'HD' not in subject_data:
        raise KeyError("A condição 'HD' não existe em subject_data.")

    df = subject_data['HD']['data']
    fs = subject_data['HD']['params']['sampling rate']
    force = df[FORCE_COLUMN].to_numpy()

    # baseline = valor mais frequente do sinal (arredondado para absorver ruído)
    rounded = np.round(force, decimals=1)
    mode_result = stats.mode(rounded, keepdims=False)
    baseline = float(np.ravel(mode_result.mode)[0])

    global_peak_value = float(np.max(force))
    threshold = baseline + 0.05 * (global_peak_value - baseline)
    min_prominence = min_prominence_ratio * (global_peak_value - baseline)
    min_distance = max(1, int(min_rep_distance_s * fs))

    peak_indices, _ = find_peaks(
        force,
        height=threshold,
        prominence=min_prominence,
        distance=min_distance,
    )

    if len(peak_indices) == 0:
        raise ValueError("Não foi detetado nenhum pico acima do threshold definido.")

    phases_list = []
    n_samples = len(force)

    for rep_i, peak_idx in enumerate(peak_indices):
        peak_value = float(force[peak_idx])
        prev_bound = phases_list[-1]['eccentric_end'] if phases_list else 0
        next_peak = peak_indices[rep_i + 1] if rep_i + 1 < len(peak_indices) else n_samples - 1

        concentric_start = prev_bound
        for i in range(peak_idx, prev_bound - 1, -1):
            if force[i] <= threshold:
                concentric_start = i
                break

        eccentric_end = next_peak
        for i in range(peak_idx, next_peak + 1):
            if force[i] <= threshold:
                eccentric_end = i
                break

        phases_list.append({
            'rep': rep_i + 1,
            'concentric_start': int(concentric_start),
            'concentric_end': int(peak_idx),
            'eccentric_start': int(peak_idx),
            'eccentric_end': int(eccentric_end),
            'baseline': baseline,
            'peak_value': peak_value,
            'fs': fs,
        })


    df['Phase'] = 'R'

    for phase in phases_list:
        rep_num = phase['rep']
        c_start, c_end = phase['concentric_start'], phase['concentric_end']
        e_start, e_end = phase['eccentric_start'], phase['eccentric_end']

        df.iloc[c_start:c_end, df.columns.get_loc('Phase')] = f'C{rep_num}'
        df.iloc[e_start:e_end + 1, df.columns.get_loc('Phase')] = f'E{rep_num}'

    subject_data['HD']['data'] = df

    if show_plot:
        _plot_phases(df, phases_list)

    return subject_data


def _plot_phases(df, phases_list):
    """
    Plots all channels of the DataFrame with the concentric and eccentric phases of every
    detected repetition shaded, marking the start/end points of each phase.

    :param df: DataFrame with all sensor channels (condition HD).
    :param phases_list: list of dicts returned by detect_concentric_eccentric_phases.
    :return: None
    """
    channels = df.columns
    n = len(channels)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.8 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, ch in zip(axes, channels):
        signal = df[ch].to_numpy()
        ax.plot(signal, color='black', linewidth=0.8)

        for rep_i, phases in enumerate(phases_list):
            cs, ce = phases['concentric_start'], phases['concentric_end']
            es, ee = phases['eccentric_start'], phases['eccentric_end']

            # só mete legenda na primeira repetição, para não repetir entradas
            label_conc = 'Fase concêntrica' if rep_i == 0 else None
            label_ecc = 'Fase excêntrica' if rep_i == 0 else None
            label_start = 'Início concêntrica' if rep_i == 0 else None
            label_peak = 'Pico' if rep_i == 0 else None
            label_end = 'Fim excêntrica' if rep_i == 0 else None

            ax.axvspan(cs, ce, color='tab:blue', alpha=0.2, label=label_conc)
            ax.axvspan(es, ee, color='tab:orange', alpha=0.2, label=label_ecc)

            ax.plot(cs, signal[cs], 'go', markersize=6, label=label_start)
            ax.plot(ce, signal[ce], 'ro', markersize=6, label=label_peak)
            ax.plot(ee, signal[ee], 'ms', markersize=6, label=label_end)

        ax.set_ylabel(ch, fontsize=8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=8)
    axes[-1].set_xlabel('Amostras')
    fig.suptitle('Fases Concêntrica e Excêntrica — Condição HD (todas as repetições)')
    plt.tight_layout()
    plt.show()

