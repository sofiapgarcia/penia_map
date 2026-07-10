import numpy as np
from matplotlib import pyplot as plt


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