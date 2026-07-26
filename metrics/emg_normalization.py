import numpy as np
from matplotlib import pyplot as plt
from constants import EMG_CHANNELS

_MVC_KEY_BY_CHANNEL = {
    "EMG_Extensor": "extensor",
    "EMG_Flexor": "flexor",
}


def get_mvc_values(subject_data, show_plot=True):
    """
    Compute the MVC (Maximum Voluntary Contraction) value for each muscle
    and store them in a new 'MVC_values' key in subject_data. The extensor
    MVC is taken as the maximum of the EMG_Extensor envelope in the HI
    condition, and the flexor MVC is taken as the maximum of the
    EMG_Flexor envelope in the MVC condition.

    Expects envelope columns (e.g. 'EMG_Flexor_envelope') to already exist
    in the respective DataFrames, as produced by get_emg_envelope().

    :param subject_data: Dict with keys 'potentiometer_position',
        'HI_values', 'HD', 'HI', 'MVC', 'power_info'. 'HD', 'HI', 'MVC' are
        each dicts with 'data' (DataFrame) and 'params' (dict)
    :param show_plot: If True, shows the envelope signal with the MVC peak
        marked, for each muscle
    :return: subject_data with 'MVC_values' key added, a dict with
        'extensor' and 'flexor' keys holding the respective MVC values
    """

    extensor_df = subject_data["HI"]["data"]
    flexor_df = subject_data["MVC"]["data"]

    extensor_envelope = extensor_df["EMG_Extensor_envelope"].values
    flexor_envelope = flexor_df["EMG_Flexor_envelope"].values

    extensor_mvc = np.max(extensor_envelope)
    flexor_mvc = np.max(flexor_envelope)

    subject_data["MVC_values"] = {
        "extensor": extensor_mvc,
        "flexor": flexor_mvc,
    }

    if show_plot:
        _plot_mvc_peak("Extensor MVC (HI condition)", extensor_envelope, extensor_mvc)
        _plot_mvc_peak("Flexor MVC (MVC condition)", flexor_envelope, flexor_mvc)

    return subject_data

def _plot_mvc_peak(title, envelope, mvc_value):
    """
    Plot an EMG envelope signal with its maximum (MVC) value marked.

    :param title: Title for the plot (e.g. "Extensor MVC (HI condition)")
    :param envelope: Envelope signal (1D array)
    :param mvc_value: MVC value (peak of the envelope) to highlight
    """

    peak_idx = np.argmax(envelope)

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(envelope, linewidth=0.8, label="Envelope")
    ax.scatter(peak_idx, mvc_value, color="red", zorder=5, label=f"MVC = {mvc_value:.4f}")
    ax.axhline(mvc_value, color="red", linestyle="--", linewidth=0.7, alpha=0.6)

    ax.set_title(title)
    ax.set_xlabel("Samples")
    ax.legend()

    plt.tight_layout()
    plt.show()


def normalize_emg(subject_data, show_plot=True):
    """
    Normalize EMG signals in HD and HI conditions as a percentage of the
    corresponding MVC (Maximal Voluntary Contraction) value stored in
    subject_data['MVC_values'] (%MVC normalization).

    Two normalized signals are produced per channel:
      - The envelope signal, divided by the matching MVC reference value
        and multiplied by 100, stored as '{channel}_norm' (e.g.
        'EMG_Flexor_norm') — same column name as before.
      - The filtered signal, divided by the matching MVC reference value
        and multiplied by 100, stored as '{channel}_filt_norm' (e.g.
        'EMG_Flexor_filt_norm').

    Both are added as new columns to the HD and HI DataFrames. The MVC
    condition is not normalized.

    Expects envelope columns (e.g. 'EMG_Flexor_envelope') and filtered
    columns (e.g. 'EMG_Flexor_filt') to already exist in the HD and HI
    DataFrames, and 'MVC_values' to already exist in subject_data (as
    produced by get_mvc_values()).

    :param subject_data: Dict with keys 'potentiometer_position',
        'HI_values', 'HD', 'HI', 'MVC', 'power_info', 'MVC_values'.
        'HD', 'HI', 'MVC' are each dicts with 'data' (DataFrame) and
        'params' (dict)
    :param show_plot: If True, shows a before/after comparison plot (all
        channels together, envelope signal) for each of HD and HI
    :return: subject_data with normalized envelope ('{channel}_norm') and
        normalized filtered ('{channel}_filt_norm') %MVC columns added to
        HD and HI DataFrames
    """

    if "MVC_values" not in subject_data:
        raise ValueError("subject_data['MVC_values'] not found — run get_mvc_values() first.")

    mvc_values = subject_data["MVC_values"]

    for condition in ("HD", "HI"):

        if condition not in subject_data:
            continue

        df = subject_data[condition]["data"]

        # Collect signals to be able to plot all channels together,
        # before vs after normalization
        envelope_signals = {}
        env_normalized_signals = {}
        filt_normalized_signals = {}

        for channel in EMG_CHANNELS:

            envelope_col = f"{channel}_envelope"
            filt_col = f"{channel}_filt"
            mvc_key = _MVC_KEY_BY_CHANNEL.get(channel)

            if envelope_col not in df.columns or mvc_key is None or mvc_key not in mvc_values:
                continue

            mvc_reference = mvc_values[mvc_key]

            envelope = df[envelope_col].values
            env_normalized = (envelope / mvc_reference) * 100

            df[f"{channel}_norm"] = env_normalized

            envelope_signals[channel] = envelope
            env_normalized_signals[channel] = env_normalized

            if filt_col in df.columns:
                filt = df[filt_col].values
                filt_normalized = (filt / mvc_reference) * 100
                df[f"{channel}_filt_norm"] = filt_normalized

                filt_normalized_signals[channel] = filt_normalized

        if show_plot and env_normalized_signals:
            _plot_emg_normalized(
                condition, envelope_signals, env_normalized_signals, filt_normalized_signals
            )

    return subject_data

def _plot_emg_normalized(condition, envelope_signals, env_normalized_signals, filt_normalized_signals=None):
    """
    Plot envelope vs normalized (%MVC) EMG signals stacked vertically, with
    all channels overlaid on each subplot for comparison. The "after"
    subplot shows both the normalized envelope and the normalized
    filtered EMG signal for each channel.

    :param condition: Condition name (e.g. "HD", "HI")
    :param envelope_signals: Dict mapping channel name to envelope signal
        (1D array), pre-normalization
    :param env_normalized_signals: Dict mapping channel name to normalized
        envelope signal (1D array), in %MVC
    :param filt_normalized_signals: Optional dict mapping channel name to
        normalized filtered EMG signal (1D array), in %MVC. Channels
        without a filtered signal are simply skipped.
    """

    filt_normalized_signals = filt_normalized_signals or {}

    fig, (ax_before, ax_after) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for channel, signal in envelope_signals.items():
        ax_before.plot(signal, linewidth=0.7, alpha=0.8, label=channel)

    ax_before.set_title(f"{condition} - Before normalization (envelope)")
    ax_before.set_ylabel("mV")
    ax_before.legend()


    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (channel, signal) in enumerate(env_normalized_signals.items()):
        color = color_cycle[i % len(color_cycle)]

        ax_after.plot(
            signal, linewidth=1.2, alpha=0.9, color=color, label=f"{channel} (envelope)"
        )

        filt_signal = filt_normalized_signals.get(channel)
        if filt_signal is not None:
            ax_after.plot(
                filt_signal,
                linewidth=0.5,
                alpha=0.4,
                color=color,
                linestyle="-",
                label=f"{channel} (EMG filtered)",
            )

    ax_after.set_title(f"{condition} - After normalization (%MVC)")
    ax_after.set_xlabel("Samples")
    ax_after.set_ylabel("% MVC")
    ax_after.legend()

    fig.suptitle(f"EMG Normalization - {condition}")
    plt.tight_layout()
    plt.show()