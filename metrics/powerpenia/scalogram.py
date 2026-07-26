import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def compute_scalogram(signal, fs, wavelet="morl", num_scales=64, min_scale=1, max_scale=128, scale_spacing="log"):
    """
    Compute the continuous wavelet transform (scalogram) of a 1D signal.

    :param signal: 1D array-like signal
    :param fs: Sampling frequency (Hz)
    :param wavelet: Wavelet name (default 'morl', Morlet)
    :param num_scales: Number of scales to use
    :param min_scale: Minimum scale value
    :param max_scale: Maximum scale value
    :param scale_spacing: 'log' (default) or 'linear'. Since frequency is
        roughly inversely proportional to scale, linear spacing bunches
        almost all the resolution into the low-frequency end and leaves the
        higher frequencies undersampled. 'log' spreads the scales evenly
        across the whole frequency band, which is what you want for EMG,
        where relevant content spans tens to hundreds of Hz.
    :return: Tuple (power, frequencies) -- power is a 2D array (scales x samples),
        frequencies is a 1D array (Hz) with one entry per scale, matching the
        rows of power.
    """
    signal = np.asarray(signal, dtype=float)

    if scale_spacing == "log":
        scales = np.geomspace(min_scale, max_scale, num_scales)
    else:
        scales = np.linspace(min_scale, max_scale, num_scales)

    coefficients, frequencies = pywt.cwt(signal, scales, wavelet, sampling_period=1 / fs)
    power = np.abs(coefficients) ** 2

    return power, frequencies


def get_median_frequency(power, frequencies):
    """
    Median frequency (Hz) of a power spectrum: the frequency that splits the
    total spectral power (averaged over time/samples) into two equal halves.

    :param power: 2D array (scales x samples) as returned by compute_scalogram
    :param frequencies: 1D array (Hz), one entry per scale/row of power
    :return: Median frequency in Hz
    """
    mean_power_per_freq = power.mean(axis=1)

    sort_idx = np.argsort(frequencies)
    freqs_sorted = frequencies[sort_idx]
    power_sorted = mean_power_per_freq[sort_idx]

    cumulative_power = np.cumsum(power_sorted)
    total_power = cumulative_power[-1]
    median_idx = np.searchsorted(cumulative_power, total_power / 2)
    median_idx = min(median_idx, len(freqs_sorted) - 1)

    return float(freqs_sorted[median_idx])


def get_peak_frequency(power, frequencies):
    """
    Peak frequency (Hz): frequency at which the spectral power (averaged over
    time/samples) is maximum.

    :param power: 2D array (scales x samples) as returned by compute_scalogram
    :param frequencies: 1D array (Hz), one entry per scale/row of power
    :return: Peak frequency in Hz
    """
    mean_power_per_freq = power.mean(axis=1)
    peak_idx = np.argmax(mean_power_per_freq)

    return float(frequencies[peak_idx])


def get_time_to_peak_energy(power, fs):
    """
    Time (s) between the start of the analysis window and the instant at
    which the total signal energy (power summed across all scales/
    frequencies at each time sample) is maximum.

    When the window passed to compute_scalogram starts at the muscle's
    onset (as is the case for both HD and HI in get_muscle_scalogram_info),
    this directly gives the time-to-peak-energy relative to onset.

    :param power: 2D array (scales x samples) as returned by compute_scalogram
    :param fs: Sampling frequency (Hz)
    :return: Time to peak energy, in seconds, relative to the first sample of power
    """
    energy_per_sample = power.sum(axis=0)
    peak_sample_idx = np.argmax(energy_per_sample)

    return float(peak_sample_idx / fs)


def get_scalogram_metrics_in_phase(
    df,
    phase_start_idx,
    phase_end_idx,
    signal_column,
    fs,
    wavelet="morl",
    num_scales=64,
    min_scale=1,
    max_scale=128,
):
    """
    Compute scalogram-derived metrics (median frequency, peak frequency, and
    time to peak energy) for one signal, within a given sample window
    (typically starting at the muscle's onset).

    :param df: DataFrame containing the signal
    :param phase_start_idx: Start sample index of the phase window (the
        muscle's onset, when used via get_muscle_scalogram_info)
    :param phase_end_idx: End sample index of the phase window
    :param signal_column: Column name of the signal to analyze
    :param fs: Sampling frequency (Hz)
    :param wavelet: Wavelet name used for the CWT (default 'morl')
    :param num_scales: Number of scales used for the CWT
    :param min_scale: Minimum scale value
    :param max_scale: Maximum scale value
    :return: Tuple (median_freq, peak_freq, time_to_peak_energy) -- the first
        two in Hz, the last in seconds since phase_start_idx (i.e. since
        onset, when phase_start_idx is the onset index)
    """
    phase_signal = df[signal_column].iloc[phase_start_idx: phase_end_idx + 1].to_numpy()

    power, frequencies = compute_scalogram(
        phase_signal, fs, wavelet=wavelet, num_scales=num_scales, min_scale=min_scale, max_scale=max_scale
    )

    median_freq = get_median_frequency(power, frequencies)
    peak_freq = get_peak_frequency(power, frequencies)
    time_to_peak_energy = get_time_to_peak_energy(power, fs)

    return median_freq, peak_freq, time_to_peak_energy


def _compute_scalogram_info_for_condition(
    df_normalized,
    onset_df,
    fs,
    onset_extensor_column,
    onset_flexor_column,
    emg_extensor_column,
    emg_flexor_column,
    wavelet,
    num_scales,
    min_scale,
    max_scale,
    start_idx_column="concentric_start_idx",
    end_idx_column="concentric_end_idx",
    rep_at_peak_power=None,
):
    """
    Compute per-repetition scalogram metrics for a single condition's onset
    DataFrame. If rep_at_peak_power is provided, the matching repetition is
    keyed as 'rep_peakpower' (used for HD); if None, all repetitions keep
    their plain numeric key (used for HI, which has no peak-power repetition).

    The analysis window boundaries are read from start_idx_column/end_idx_column,
    independently for each muscle's row. For HD this is the shared concentric
    phase ('concentric_start_idx'/'concentric_end_idx'); for HI (isometric,
    no concentric/eccentric phases) this should instead be the muscle's own
    activation window ('onset_idx'/'offset_idx'), which can differ slightly
    between the extensor and flexor since each envelope is computed independently.

    :param df_normalized: DataFrame containing the normalized EMG signals
    :param onset_df: DataFrame output for the condition from detect_emg_onsets
    :param fs: Sampling frequency (Hz)
    :param onset_extensor_column: Muscle column name used for extensor onsets
    :param onset_flexor_column: Muscle column name used for flexor onsets
    :param emg_extensor_column: Column name of normalized extensor EMG
    :param emg_flexor_column: Column name of normalized flexor EMG
    :param wavelet: Wavelet name used for the CWT
    :param num_scales: Number of scales used for the CWT
    :param min_scale: Minimum scale value
    :param max_scale: Maximum scale value
    :param start_idx_column: Column in onset_df with the window start index
    :param end_idx_column: Column in onset_df with the window end index
    :param rep_at_peak_power: Repetition number to key as 'rep_peakpower', or None
    :return: Dict keyed by repetition (int or 'rep_peakpower') with metric dicts.
        Each metric dict includes median/peak frequency (and their ratio) plus
        'time_to_peak_energy_extensor'/'time_to_peak_energy_flexor' -- the time
        (s) between the window start (start_idx_column, i.e. the muscle's
        onset) and the instant of peak signal energy.
    """
    scalogram_info = {}

    # Group rows by repetition number from the onset DataFrame
    repetition_groups = onset_df.groupby("rep")

    for rep, rep_data in repetition_groups:
        ext_row = rep_data[rep_data["muscle"] == onset_extensor_column]
        flex_row = rep_data[rep_data["muscle"] == onset_flexor_column]

        if ext_row.empty or flex_row.empty:
            continue

        # Extract the window boundaries independently per muscle (they coincide
        # for HD, since the concentric phase is shared, but can differ for HI,
        # since each muscle's onset/offset comes from its own envelope)
        extensor_start_idx = ext_row[start_idx_column].values[0]
        extensor_end_idx = ext_row[end_idx_column].values[0]
        flexor_start_idx = flex_row[start_idx_column].values[0]
        flexor_end_idx = flex_row[end_idx_column].values[0]

        if pd.isna(extensor_start_idx) or pd.isna(extensor_end_idx) or \
                pd.isna(flexor_start_idx) or pd.isna(flexor_end_idx):
            continue

        median_freq_extensor, peak_freq_extensor, time_to_peak_extensor = get_scalogram_metrics_in_phase(
            df_normalized,
            int(extensor_start_idx),
            int(extensor_end_idx),
            emg_extensor_column,
            fs,
            wavelet=wavelet,
            num_scales=num_scales,
            min_scale=min_scale,
            max_scale=max_scale,
        )
        median_freq_flexor, peak_freq_flexor, time_to_peak_flexor = get_scalogram_metrics_in_phase(
            df_normalized,
            int(flexor_start_idx),
            int(flexor_end_idx),
            emg_flexor_column,
            fs,
            wavelet=wavelet,
            num_scales=num_scales,
            min_scale=min_scale,
            max_scale=max_scale,
        )

        median_freq_ratio = median_freq_extensor / median_freq_flexor
        peak_freq_ratio = peak_freq_extensor / peak_freq_flexor

        if rep_at_peak_power is not None and rep == rep_at_peak_power:
            rep_key = "rep_peakpower"
        else:
            rep_key = rep

        scalogram_info[rep_key] = {
            "median_freq_extensor": median_freq_extensor,
            "median_freq_flexor": median_freq_flexor,
            "median_freq_ratio": median_freq_ratio,
            "peak_freq_extensor": peak_freq_extensor,
            "peak_freq_flexor": peak_freq_flexor,
            "peak_freq_ratio": peak_freq_ratio,
            "time_to_peak_energy_extensor": time_to_peak_extensor,
            "time_to_peak_energy_flexor": time_to_peak_flexor,
        }

    return scalogram_info


def get_muscle_scalogram_info(
    subject_data_normalized,
    onset_df_hd,
    onset_df_hi,
    power_info,
    subject_id,
    fs=None,
    onset_extensor_column="EMG_Extensor_filt",
    onset_flexor_column="EMG_Flexor_filt",
    emg_extensor_column="EMG_Extensor_filt_norm",
    emg_flexor_column="EMG_Flexor_filt_norm",
    wavelet="morl",
    num_scales=256,
    min_scale=1,
    max_scale=256,
    show_scalogram_plot=False,
    scalogram_plot_show_signal=True,
    scalogram_plot_cmap="viridis",
):
    """
    Build the full muscle scalogram (spectral) info for both HD and HI
    conditions using their respective onset DataFrames from
    detect_emg_onsets. For each repetition, computes the scalogram of the
    extensor and flexor EMG within an analysis window and derives the median
    frequency, peak frequency, and time to peak energy for each muscle.

    The analysis window always starts at the muscle's own onset
    ('onset_idx'). It ends at the concentric phase end
    ('concentric_end_idx') for HD, and at the muscle's own offset
    ('offset_idx') for HI (isometric condition, no concentric/eccentric
    phases) -- see _compute_scalogram_info_for_condition.

    The scalogram is computed on the normalized filtered EMG signal
    ('EMG_Extensor_filt_norm' / 'EMG_Flexor_filt_norm' by default), not on
    the normalized envelope, since the envelope's smoothing removes most of
    the spectral content that the CWT is meant to characterize.

    HD has a repetition of peak power (identified via power_info and keyed
    as 'rep_peakpower' in the output); HI has no such repetition, so all of
    its repetitions keep their plain numeric key.

    :param subject_data_normalized: Dict output of normalize_emg containing 'data' and 'params' for both conditions ('HD' and 'HI' keys)
    :param onset_df_hd: DataFrame output for condition 'HD' from detect_emg_onsets
    :param onset_df_hi: DataFrame output for condition 'HI' from detect_emg_onsets
    :param power_info: DataFrame containing a 'rep_at_peak_power' column (HD only)
    :param subject_id: Subject identifier tag
    :param fs: Sampling frequency (Hz). If None, extracted per-condition from subject_data_normalized
    :param onset_extensor_column: Muscle column name used for extensor onsets
    :param onset_flexor_column: Muscle column name used for flexor onsets
    :param emg_extensor_column: Column name of the normalized (filtered) extensor EMG
    :param emg_flexor_column: Column name of the normalized (filtered) flexor EMG
    :param wavelet: Wavelet name used for the CWT (default 'morl')
    :param num_scales: Number of scales used for the CWT
    :param min_scale: Minimum scale value
    :param max_scale: Maximum scale value
    :param show_scalogram_plot: If True, additionally plots the scalogram of the
        entire EMG signal (extensor and flexor, HD and HI) via
        `plot_condition_muscle_scalograms`, in addition to computing the metrics.
        The analysis window of each repetition (onset -> concentric end for
        HD, onset -> offset for HI) is shaded on both panels using
        onset_df_hd / onset_df_hi.
    :param scalogram_plot_show_signal: If show_scalogram_plot is True, whether to
        include a panel with the raw EMG signal above each scalogram plot
    :param scalogram_plot_cmap: If show_scalogram_plot is True, the colormap used
        for the scalogram heatmaps
    :return: Tuple (df_hd, df_hi) containing scalogram DataFrames
    """
    rep_at_peak_power = int(power_info["rep_at_peak_power"].iloc[0])

    fs_hd = fs if fs is not None else subject_data_normalized["HD"]["params"]["sampling rate"]
    fs_hi = fs if fs is not None else subject_data_normalized["HI"]["params"]["sampling rate"]

    scalogram_info_hd = _compute_scalogram_info_for_condition(
        subject_data_normalized["HD"]["data"],
        onset_df_hd,
        fs_hd,
        onset_extensor_column,
        onset_flexor_column,
        emg_extensor_column,
        emg_flexor_column,
        wavelet,
        num_scales,
        min_scale,
        max_scale,
        start_idx_column="onset_idx",
        end_idx_column="concentric_end_idx",
        rep_at_peak_power=rep_at_peak_power,
    )

    scalogram_info_hi = _compute_scalogram_info_for_condition(
        subject_data_normalized["HI"]["data"],
        onset_df_hi,
        fs_hi,
        onset_extensor_column,
        onset_flexor_column,
        emg_extensor_column,
        emg_flexor_column,
        wavelet,
        num_scales,
        min_scale,
        max_scale,
        start_idx_column="onset_idx",
        end_idx_column="offset_idx",
        rep_at_peak_power=None,
    )

    df_hd = build_scalogram_dataframe({"HD": scalogram_info_hd}, subject_id, condition="HD")
    df_hi = build_scalogram_dataframe({"HI": scalogram_info_hi}, subject_id, condition="HI")

    if show_scalogram_plot:
        plot_condition_muscle_scalograms(
            subject_data_normalized,
            "HD",
            onset_df=onset_df_hd,
            onset_extensor_column=onset_extensor_column,
            onset_flexor_column=onset_flexor_column,
            emg_extensor_column=emg_extensor_column,
            emg_flexor_column=emg_flexor_column,
            start_idx_column="onset_idx",
            end_idx_column="concentric_end_idx",
            fs=fs_hd,
            wavelet=wavelet,
            num_scales=num_scales,
            min_scale=min_scale,
            max_scale=max_scale,
            show_signal=scalogram_plot_show_signal,
            cmap=scalogram_plot_cmap,
        )
        plot_condition_muscle_scalograms(
            subject_data_normalized,
            "HI",
            onset_df=onset_df_hi,
            onset_extensor_column=onset_extensor_column,
            onset_flexor_column=onset_flexor_column,
            emg_extensor_column=emg_extensor_column,
            emg_flexor_column=emg_flexor_column,
            start_idx_column="onset_idx",
            end_idx_column="offset_idx",
            fs=fs_hi,
            wavelet=wavelet,
            num_scales=num_scales,
            min_scale=min_scale,
            max_scale=max_scale,
            show_signal=scalogram_plot_show_signal,
            cmap=scalogram_plot_cmap,
        )

    return df_hd, df_hi


def build_scalogram_dataframe(scalogram_info, subject_id=None, condition="HD"):
    """
    Flatten scalogram_info into a tidy DataFrame with one row per repetition.

    :param scalogram_info: Dict containing metric values per repetition key
    :param subject_id: Optional subject identifier to prepend as first column
    :param condition: Condition key inside scalogram_info (default 'HD')
    :return: Tidy DataFrame with repetition metrics
    """
    rows = []
    for rep_key, metrics in scalogram_info[condition].items():
        row = {"repetition": rep_key}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)

    if subject_id is not None:
        df.insert(0, "subject_id", subject_id)

    return df


def plot_signal_scalogram(
    df,
    signal_column,
    fs,
    wavelet="morl",
    num_scales=64,
    min_scale=1,
    max_scale=128,
    scale_spacing="log",
    show_signal=True,
    cmap="viridis",
    title=None,
    freq_max=450,
    color_scale="log",
    phase_windows=None,
    phase_color="white",
    phase_alpha=0.22,
    show_phase_labels=True,
):
    """
    Compute and plot the scalogram (time-frequency power) of an entire EMG
    signal -- i.e. not restricted to any repetition/phase window, the full
    recording for that column.

    :param df: DataFrame containing the signal
    :param signal_column: Column name of the signal to analyze (e.g. an
        'EMG_Extensor_filt_norm' or 'EMG_Extensor_filt' column)
    :param fs: Sampling frequency (Hz)
    :param wavelet: Wavelet name used for the CWT (default 'morl')
    :param num_scales: Number of scales used for the CWT
    :param min_scale: Minimum scale value
    :param max_scale: Maximum scale value
    :param scale_spacing: 'log' (default) or 'linear', passed to
        compute_scalogram. 'log' spreads resolution evenly across the whole
        frequency band instead of bunching it at low frequencies.
    :param show_signal: If True, adds a panel above the scalogram with the raw
        EMG signal itself, sharing the time axis
    :param cmap: Matplotlib colormap used for the scalogram heatmap
    :param title: Optional custom title. Defaults to a title built from signal_column
    :param freq_max: Upper limit (Hz) for the y-axis. EMG spectral content is
        essentially all below ~450-500 Hz, so plotting up to the Nyquist
        frequency just wastes most of the plot on an empty region and makes
        the low-frequency band look compressed. Set to None to show the full
        computed range instead.
    :param color_scale: 'log' (default) or 'linear'. EMG scalogram power is
        typically dominated by a few very large low-frequency values, which
        saturates a linear color scale and makes everything else look flat.
        'log' compresses that range so weaker, higher-frequency activity
        becomes visible too.
    :param phase_windows: Optional list of (rep, start_idx, end_idx) tuples
        marking the contraction window of each repetition (concentric phase
        for HD, onset-offset window for HI). Each window is shaded across
        both the signal and scalogram panels (sample indices converted to
        seconds via fs) so the contraction period is visible directly on
        the plot.
    :param phase_color: Shading/line color used for phase_windows
    :param phase_alpha: Alpha (opacity) of the shaded phase_windows regions
    :param show_phase_labels: If True and phase_windows is given, annotates
        each shaded region with its repetition number near the top of the
        scalogram panel
    :return: None. Displays the plot.
    """
    signal = df[signal_column].to_numpy()
    power, frequencies = compute_scalogram(
        signal,
        fs,
        wavelet=wavelet,
        num_scales=num_scales,
        min_scale=min_scale,
        max_scale=max_scale,
        scale_spacing=scale_spacing,
    )

    print(f"frequencia minima: {frequencies.min()}, frequencia maxima: {frequencies.max()}")

    time = np.arange(signal.size) / fs

    if show_signal:
        fig, (ax_signal, ax_scalogram) = plt.subplots(
            2, 1, figsize=(14, 6), sharex=True, gridspec_kw={"height_ratios": [1, 3]}
        )
        ax_signal.plot(time, signal, color="black", linewidth=0.6)
        ax_signal.set_ylabel(signal_column, fontsize=8)
    else:
        fig, ax_scalogram = plt.subplots(figsize=(14, 5))
        ax_signal = None

    if color_scale == "log":
        # Evita log(0): usa o menor valor positivo de power como piso
        positive_power = power[power > 0]
        vmin = positive_power.min() if positive_power.size else 1e-12
        norm = LogNorm(vmin=vmin, vmax=power.max())
        mesh = ax_scalogram.pcolormesh(time, frequencies, power, shading="auto", cmap=cmap, norm=norm)
    else:
        mesh = ax_scalogram.pcolormesh(time, frequencies, power, shading="auto", cmap=cmap)

    ax_scalogram.set_ylabel("Frequency (Hz)")
    ax_scalogram.set_xlabel("Time (s)")
    if freq_max is not None:
        ax_scalogram.set_ylim(frequencies.min(), freq_max)
    fig.colorbar(mesh, ax=ax_scalogram, label="Power")

    # Assinala a janela de contração (fase concêntrica em HD, onset-offset em HI)
    # de cada repetição, tanto no painel do sinal como no do escalograma.
    if phase_windows:
        y_top = freq_max if freq_max is not None else frequencies.max()
        label_y = frequencies.min() + 0.95 * (y_top - frequencies.min())

        for rep, start_idx, end_idx in phase_windows:
            start_t = start_idx / fs
            end_t = end_idx / fs

            ax_scalogram.axvspan(start_t, end_t, color=phase_color, alpha=phase_alpha, linewidth=0, zorder=3)
            ax_scalogram.axvline(start_t, color=phase_color, linewidth=0.8, linestyle="--", zorder=4)
            ax_scalogram.axvline(end_t, color=phase_color, linewidth=0.8, linestyle="--", zorder=4)

            if show_phase_labels:
                ax_scalogram.text(
                    (start_t + end_t) / 2,
                    label_y,
                    f"rep {rep}",
                    ha="center",
                    va="top",
                    fontsize=7,
                    color=phase_color,
                    zorder=5,
                )

            if ax_signal is not None:
                ax_signal.axvspan(start_t, end_t, color=phase_color, alpha=phase_alpha, linewidth=0, zorder=3)
                ax_signal.axvline(start_t, color="black", linewidth=0.6, linestyle="--", alpha=0.6, zorder=4)
                ax_signal.axvline(end_t, color="black", linewidth=0.6, linestyle="--", alpha=0.6, zorder=4)

    fig.suptitle(title if title is not None else f"Scalogram -- {signal_column}")
    plt.tight_layout()
    plt.show()


def plot_condition_muscle_scalograms(
    subject_data_normalized,
    condition,
    onset_df=None,
    onset_extensor_column="EMG_Extensor_filt",
    onset_flexor_column="EMG_Flexor_filt",
    emg_extensor_column="EMG_Extensor_filt_norm",
    emg_flexor_column="EMG_Flexor_filt_norm",
    start_idx_column="concentric_start_idx",
    end_idx_column="concentric_end_idx",
    fs=None,
    wavelet="morl",
    num_scales=64,
    min_scale=1,
    max_scale=128,
    scale_spacing="log",
    show_signal=True,
    cmap="viridis",
    freq_max=450,
    color_scale="log",
    phase_color="white",
    phase_alpha=0.22,
    show_phase_labels=True,
):
    """
    Convenience wrapper around `plot_signal_scalogram` that plots the
    full-signal scalogram for both the extensor and the flexor EMG of a
    given condition (e.g. 'HD' or 'HI'), one figure per muscle.

    If onset_df is given (the condition's onset DataFrame from
    detect_emg_onsets), the contraction window of every repetition is shaded
    on each plot: for HD this is the shared concentric phase
    (start_idx_column/end_idx_column default to 'concentric_start_idx'/
    'concentric_end_idx'), for HI it's each muscle's own activation window
    (pass start_idx_column='onset_idx', end_idx_column='offset_idx'), which
    can differ slightly between extensor and flexor.

    :param subject_data_normalized: Dict output of normalize_emg containing 'data' and 'params'
    :param condition: Condition name to plot (e.g. 'HD' or 'HI')
    :param onset_df: Optional onset DataFrame for this condition (onset_df_hd
        or onset_df_hi) from detect_emg_onsets, used to shade the contraction
        window of each repetition. If None, no shading is drawn.
    :param onset_extensor_column: Muscle name (as it appears in onset_df's
        'muscle' column) used to select the extensor's rows
    :param onset_flexor_column: Muscle name (as it appears in onset_df's
        'muscle' column) used to select the flexor's rows
    :param emg_extensor_column: Column name of the normalized (filtered) extensor EMG to plot
    :param emg_flexor_column: Column name of the normalized (filtered) flexor EMG to plot
    :param start_idx_column: Column in onset_df with the window start index
        ('concentric_start_idx' for HD, 'onset_idx' for HI)
    :param end_idx_column: Column in onset_df with the window end index
        ('concentric_end_idx' for HD, 'offset_idx' for HI)
    :param fs: Sampling frequency (Hz). If None, extracted from subject_data_normalized
    :param wavelet: Wavelet name used for the CWT (default 'morl')
    :param num_scales: Number of scales used for the CWT
    :param min_scale: Minimum scale value
    :param max_scale: Maximum scale value
    :param scale_spacing: 'log' (default) or 'linear', see compute_scalogram
    :param show_signal: If True, adds a panel above each scalogram with the raw EMG signal
    :param cmap: Matplotlib colormap used for the scalogram heatmaps
    :param freq_max: Upper y-axis limit (Hz), see plot_signal_scalogram
    :param color_scale: 'log' (default) or 'linear' color mapping, see plot_signal_scalogram
    :param phase_color: Shading/line color used for the contraction window
    :param phase_alpha: Alpha (opacity) of the shaded contraction window
    :param show_phase_labels: If True, annotates each shaded window with its
        repetition number
    :return: None. Displays two plots (one per muscle).
    """
    if fs is None:
        fs = subject_data_normalized[condition]["params"]["sampling rate"]

    df = subject_data_normalized[condition]["data"]

    # (coluna do sinal a plotar, nome do músculo correspondente em onset_df)
    signal_muscle_pairs = (
        (emg_extensor_column, onset_extensor_column),
        (emg_flexor_column, onset_flexor_column),
    )

    for signal_column, onset_muscle_name in signal_muscle_pairs:
        phase_windows = None

        if onset_df is not None:
            muscle_rows = onset_df[onset_df["muscle"] == onset_muscle_name]
            phase_windows = [
                (row["rep"], int(row[start_idx_column]), int(row[end_idx_column]))
                for _, row in muscle_rows.iterrows()
                if pd.notna(row[start_idx_column]) and pd.notna(row[end_idx_column])
            ]

        plot_signal_scalogram(
            df,
            signal_column,
            fs,
            wavelet=wavelet,
            num_scales=num_scales,
            min_scale=min_scale,
            max_scale=max_scale,
            scale_spacing=scale_spacing,
            show_signal=show_signal,
            cmap=cmap,
            title=f"{condition} -- Scalogram -- {signal_column}",
            freq_max=freq_max,
            color_scale=color_scale,
            phase_windows=phase_windows,
            phase_color=phase_color,
            phase_alpha=phase_alpha,
            show_phase_labels=show_phase_labels,
        )