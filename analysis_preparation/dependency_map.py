# oi

def _build_prefixed_map(prefix, suffix_template):
    """
    Expand a suffix-level dependency template into full variable names
    for a given prefix.

    :param prefix: e.g. "rep_max_hi_", "mean_hi_", "rep_pp_hd_", "mean_hd_"
    :param suffix_template: dict {derived_suffix: [origin_suffix, ...]}
    :return: dict {derived_var: [origin_var, ...]} with prefix applied
    """
    return {
        f"{prefix}{derived_suffix}": [f"{prefix}{o}" for o in origin_suffixes]
        for derived_suffix, origin_suffixes in suffix_template.items()
    }


# ---------------------------------------------------------------------------
# Dynapenia
# ---------------------------------------------------------------------------
_HI_SUFFIX_TEMPLATE = {
    "onset_diff_ms_abs": ["onset_diff_ms"],
    "cocontraction_normalized": ["cocontraction_ms"],
    "emg_peak_ratio": ["emg_peak_extensor", "emg_peak_flexor"],
    "emg_overlap_area_normalized": ["emg_overlap_area"],
    "Median_Frequency_ratio": ["Median_Frequency_extensor", "Median_Frequency_flexor"],
    "Peak_Frequency_ratio": ["Peak_Frequency_extensor", "Peak_Frequency_flexor"],
}

DYNAPENIA_DEPENDENCY_MAP = {
    **_build_prefixed_map("rep_max_hi_", _HI_SUFFIX_TEMPLATE),
    **_build_prefixed_map("mean_hi_", _HI_SUFFIX_TEMPLATE),
}

# ---------------------------------------------------------------------------
# Powerpenia
# ---------------------------------------------------------------------------
_HD_SUFFIX_TEMPLATE = {
    "onset_diff_ms_abs": ["onset_diff_ms"],
    "cocontraction_normalized": ["cocontraction_ms"],
    "emg_peak_ratio": ["emg_peak_extensor", "emg_peak_flexor"],
    "envelope_peak_delay_ms_abs": ["envelope_peak_delay_ms"],
    "normalized_envelope_peak_delay": ["envelope_peak_delay_ms"],
    "normalized_envelope_peak_delay_abs": ["envelope_peak_delay_ms_abs"],
    "emg_overlap_area_normalized": ["emg_overlap_area"],
    "Median_Frequency_ratio": ["Median_Frequency_extensor", "Median_Frequency_flexor"],
    "Peak_Frequency_ratio": ["Peak_Frequency_extensor", "Peak_Frequency_flexor"],
}

POWERPENIA_DEPENDENCY_MAP = {
    **_build_prefixed_map("rep_pp_hd_", _HD_SUFFIX_TEMPLATE),
    **_build_prefixed_map("mean_hd_", _HD_SUFFIX_TEMPLATE),
    "peak_power_mass": ["peak_power","force_at_peak_power","velocity_at_peak_power"],
    "peak_power_height_squared": ["peak_power","force_at_peak_power", "velocity_at_peak_power"],
    "peak_power": ["force_at_peak_power", "velocity_at_peak_power"],
}

# ---------------------------------------------------------------------------
# Osteopenia
# ---------------------------------------------------------------------------
OSTEOPENIA_DEPENDENCY_MAP = {
    "Bone Mineral Density": ["Estimated bone mass kg"]
}

# ---------------------------------------------------------------------------
# Sarcopenia
# ---------------------------------------------------------------------------
SARCOPENIA_DEPENDENCY_MAP = {

    "ALMI": [
        "Arm muscle (right) kg",
        "Arm muscle (left) kg",
        "Leg muscle (right) kg",
        "Leg muscle (left) kg",
        "Height(cm)"
    ],
}

ANTHROPOMETRIC_METRICS = {"Body mass index (BMI)": ["Body mass(kg)","Height(cm)"]}

# ---------------------------------------------------------------------------
# Combined map
# ---------------------------------------------------------------------------
DEPENDENCY_MAP = {
    **DYNAPENIA_DEPENDENCY_MAP,
    **POWERPENIA_DEPENDENCY_MAP,
    **OSTEOPENIA_DEPENDENCY_MAP,
    **SARCOPENIA_DEPENDENCY_MAP,
    **ANTHROPOMETRIC_METRICS
}