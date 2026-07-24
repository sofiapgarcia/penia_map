"""
Muscle coordination metrics.

Computes, per repetition, the temporal coordination between the extensor and
flexor muscles (onset difference, co-contraction time) and the relative EMG
activation (normalized EMG peak ratio) during the concentric phase.

Assumptions (adjust if they don't match your pipeline):
- `subject_data_normalized[condition]["data"]` is the DataFrame that already
  went through `normalize_emg`. The normalized EMG signals live in their own
  columns, "EMG_Extensor_norm" / "EMG_Flexor_norm", separate from the raw
  filtered ones ("EMG_Extensor_filt" / "EMG_Flexor_filt") that onset_info's
  keys refer to. That's why the function takes two separate pairs of column
  names: onset_extensor_column/onset_flexor_column (for looking up onsets in
  onset_info) and emg_extensor_column/emg_flexor_column (for the actual EMG
  peak values, which must come from the normalized signal).
- The EMG peak is taken as the maximum of the normalized EMG signal within
  the concentric phase window (`concentric_phase["start"]` ->
  `concentric_phase["end"]`), since that's the "power" phase of the
  repetition. If you actually want it at a single instant (e.g. the exact
  sample of peak power), swap `get_emg_peak_in_phase` for a single-sample
  lookup instead.
- Currently only implemented for the "HD" condition, as requested, but the
  functions are condition-agnostic (they just need onset + concentric_phase
  info), so extending to other conditions later is just a matter of passing
  condition="HI" etc. and adapting the onset_info structure for that
  condition (HI currently has "contraction" instead of "concentric_phase").
"""

import pandas as pd


def get_onset_difference(onset_extensor_idx, onset_flexor_idx, fs):
    """
    Difference (ms) between the extensor and flexor onset.
    Positive -> extensor onset happened AFTER flexor onset.
    Negative -> extensor onset happened BEFORE flexor onset.
    """
    diff_samples = onset_extensor_idx - onset_flexor_idx
    return (diff_samples / fs) * 1000


def get_onset_difference_abs(onset_extensor_idx, onset_flexor_idx, fs):
    """
    Absolute value (ms) of the onset difference between extensor and flexor.
    """
    return abs(get_onset_difference(onset_extensor_idx, onset_flexor_idx, fs))


def get_cocontraction_time(onset_extensor_idx, onset_flexor_idx, concentric_end_idx, fs):
    """
    Time (ms) between the onset of whichever muscle activated LAST and the
    end of the concentric phase.
    """
    last_onset_idx = max(onset_extensor_idx, onset_flexor_idx)
    cocontraction_samples = concentric_end_idx - last_onset_idx
    return (cocontraction_samples / fs) * 1000


def get_emg_peak_in_phase(df_normalized, phase_start_idx, phase_end_idx, emg_column):
    """
    Peak (max) value of the normalized EMG signal for one muscle, within a
    given sample window (by default the concentric phase of the repetition).
    """
    phase_signal = df_normalized[emg_column].iloc[phase_start_idx:phase_end_idx + 1]
    return phase_signal.max()


def get_emg_peak_ratio(emg_peak_extensor, emg_peak_flexor):
    """
    Ratio between extensor and flexor normalized EMG peaks:
    emg_peak_extensor / emg_peak_flexor
    """
    return emg_peak_extensor / emg_peak_flexor


def get_muscle_coordination_info(
    subject_data_normalized,
    onset_info,
    power_info,
    condition="HD",
    fs=None,
    onset_extensor_column="EMG_Extensor_filt",
    onset_flexor_column="EMG_Flexor_filt",
    emg_extensor_column="EMG_Extensor_norm",
    emg_flexor_column="EMG_Flexor_norm",
):
    """
    Build the full muscle coordination info for a given condition (only "HD"
    supported for now).

    Parameters
    ----------
    subject_data_normalized : dict
        Output of `normalize_emg` (has subject_data_normalized[condition]["data"]
        and ["params"]).
    onset_info : dict
        Output of `detect_emg_onsets`. Expects, per repetition:
        {"onset": {extensor_column: idx, flexor_column: idx},
         "concentric_phase": {"start": idx, "end": idx}}
    power_info : pandas.DataFrame
        Output row(s) from `process_hd_power`. Must contain a
        "rep_at_peak_power" column.
    condition : str
        Condition to process. Only "HD" is implemented right now.
    fs : float or None
        Sampling rate. If None, it's pulled from
        subject_data_normalized[condition]["params"]["sampling rate"].
    onset_extensor_column, onset_flexor_column : str
        Column names used as keys inside onset_info[condition][rep]["onset"]
        (these are the raw/filtered EMG column names, e.g. "EMG_Extensor_filt").
    emg_extensor_column, emg_flexor_column : str
        Column names of the NORMALIZED EMG signals in
        subject_data_normalized[condition]["data"], used to compute the EMG
        peaks (e.g. "EMG_Extensor_norm").

    Returns
    -------
    dict
        {condition: {rep_number_or_"rep_peakpower": {metrics...}, ...}}
        The repetition that matches power_info["rep_at_peak_power"] is keyed
        as "rep_peakpower" instead of its numeric repetition number.
    """
    if condition != "HD":
        raise NotImplementedError(
            "get_muscle_coordination_info currently only supports condition='HD'."
        )

    if fs is None:
        fs = subject_data_normalized[condition]["params"]["sampling rate"]

    df_normalized = subject_data_normalized[condition]["data"]
    rep_at_peak_power = int(power_info["rep_at_peak_power"].iloc[0])

    coordination_info = {}

    for rep, rep_info in onset_info[condition].items():
        onset_extensor_idx = rep_info["onset"][onset_extensor_column]
        onset_flexor_idx = rep_info["onset"][onset_flexor_column]
        concentric_start_idx = rep_info["concentric_phase"]["start"]
        concentric_end_idx = rep_info["concentric_phase"]["end"]

        onset_diff_ms = get_onset_difference(onset_extensor_idx, onset_flexor_idx, fs)
        onset_diff_ms_abs = abs(onset_diff_ms)
        cocontraction_ms = get_cocontraction_time(
            onset_extensor_idx, onset_flexor_idx, concentric_end_idx, fs
        )

        emg_peak_extensor = get_emg_peak_in_phase(
            df_normalized, concentric_start_idx, concentric_end_idx, emg_extensor_column
        )
        emg_peak_flexor = get_emg_peak_in_phase(
            df_normalized, concentric_start_idx, concentric_end_idx, emg_flexor_column
        )
        emg_peak_ratio = get_emg_peak_ratio(emg_peak_extensor, emg_peak_flexor)

        rep_key = "rep_peakpower" if rep == rep_at_peak_power else rep

        coordination_info[rep_key] = {
            "onset_diff_ms": onset_diff_ms,
            "onset_diff_ms_abs": onset_diff_ms_abs,
            "cocontraction_ms": cocontraction_ms,
            "emg_peak_extensor": emg_peak_extensor,
            "emg_peak_flexor": emg_peak_flexor,
            "emg_peak_ratio": emg_peak_ratio,
        }

    return {condition: coordination_info}


def build_coordination_dataframe(coordination_info, subject_id=None, condition="HD"):
    """
    Flatten coordination_info[condition] into a tidy DataFrame (one row per
    repetition), handy for Excel export alongside power_info. Optionally
    tags each row with subject_id, mirroring the pattern used for
    power_info in the main script.
    """
    rows = []
    for rep_key, metrics in coordination_info[condition].items():
        row = {"repetition": rep_key}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)

    if subject_id is not None:
        df.insert(0, "subject_id", subject_id)

    return df