CATEGORICAL_FEATURES = [
    "Gender",
    "IPAQ",
    "HHIE-S (Classification)",
    "Dominat Hand",
    "Marital Status",
    "Education Level",
    "Stimulus",
]

groups = {
    "dados_demográficos": [
        "Idade", "Gender_M", "Marital Status_Married", "Marital Status_Single",
        "Marital Status_União de Facto", "Marital Status_Widow",
        "Education Level_Ensino Superior", "Education Level_Pós-Graduação",
        "Education Level_Secundário"
    ],

    "antropometria": [
        "Massa Corporal", "Altura", "BMI_DXA"
    ],

    "percentagem_gordura_tecido_mole": [
        "Perc_gordura_total", "Perc_gordura_tronco", "Perc_gordura_pernas",
        "Perc_gordura_androide", "Perc_gordura_ginoide",
        "Perc_gordura_bracos_total", "Perc_gordura_braco_direito",
        "Perc_gordura_braco_esquerdo", "Perc_gordura_dif_bracos",
        "Perc_gordura_perna_direita", "Perc_gordura_perna_esquerda",
        "Perc_gordura_dif_pernas", "Perc_gordura_regiao_total"
    ],

    "massa_total": [
        "Massa_total", "Massa_tronco", "Massa_pernas",
        "Massa_total_bracos", "Massa_braco_direito", "Massa_braco_esquerdo",
        "Massa_dif_bracos", "Massa_perna_direita", "Massa_perna_esquerda",
        "Massa_dif_pernas", "Massa_tronco_direito", "Massa_tronco_esquerdo",
        "Massa_androide", "Massa_ginoide", "Tecido"
    ],

    "massa_gordura": [
        "Gordo_total",
        "Massa_gordura_bracos", "Massa_gordura_braco_direito",
        "Massa_gordura_braco_esquerdo", "Massa_gordura_dif_bracos",
        "Massa_gordura_pernas", "Massa_gordura_perna_direita",
        "Massa_gordura_perna_esquerda", "Massa_gordura_dif_pernas",
        "Massa_gordura_tronco", "Massa_gordura_androide",
        "Massa_gordura_ginoide", "VAT_massa_gordura", "TAS_massa_gordura"
    ],

    "massa_magra": [
        "Magro_total", "Magro_tronco", "Magro_pernas",
        "Massa_magra_bracos", "Massa_magra_braco_direito",
        "Massa_magra_braco_esquerdo", "Massa_magra_dif_bracos",
        "Massa_magra_perna_direita", "Massa_magra_perna_esquerda",
        "Massa_magra_dif_pernas", "Massa_magra_androide",
        "Massa_magra_ginoide", "Massa_magra_tronco_direito",
        "Massa_magra_tronco_esquerdo", "Massa_magra_tronco_dif",
        "Massa_magra_direita", "Massa_magra_esquerda",
        "Massa_magra_dif", "Massa_livre_gordura", "Perc_tecido_magro"
    ],

    "massa_ossea": [
        "BMC_total", "BMC_tronco", "BMC_pernas", "BMC_pernas.1",
        "BMC_bracos", "BMC_braco_direito", "BMC_braco_esquerdo",
        "BMC_dif_bracos", "BMC_perna_direita", "BMC_perna_esquerda",
        "BMC_dif_pernas", "BMC_tronco.1", "BMC_androide", "BMC_ginoide"
    ],

    "densidade_ossea": [
        "BMD", "BMD_cabeca", "BMD_bracos", "BMD_pernas",
        "BMD_tronco", "BMD_costelas", "BMD_coluna", "BMD_pelve",
        "T_score", "Z_score", "Centil"
    ],

    "percentagem_gordura_total": [
        "Dist_gordura_bracos", "Dist_gordura_pernas",
        "Dist_gordura_tronco", "Dist_gordura_androide",
        "Dist_gordura_ginoide"
    ],

    "gordura_visceral": [
        "VAT_volume", "VAT_area"
    ],

    "gordura_subcutanea": [
        "TAS_volume", "TAS_area"
    ],

    "metabolismo": [
        "RMR", "Daily calorie intake (DCI)", "Estimated metabolic age"
    ],

    "performance_fisica": [
        "Hand Grip_F", "1RM_BP_F", "1RM_SQ_F",
        "Peak_Power_BP_F", "Peak_Power_SQ_F"
    ],

    "atividade_fisica": [
        "MET/Week", "IPAQ_Low", "IPAQ_Moderate"
    ],

    "clinico_funcional": [
        "HHIE-S (score)", "HHIE-S (Classification)_No Handicap"
    ],

    "indices_composicao": [
        "RSMI","Taxa_A_G"
    ],

    "outros": [
        "Stimulus_GM"
    ]
}

PENIA_MAP_GROUPS = {
    "Dynapenia": [
    "coord_hi_onset_diff_ms",
    "coord_hi_onset_diff_ms_abs",
    "coord_hi_cocontraction_ms",
    "coord_hi_cocontraction_normalized",
    "coord_hi_emg_peak_extensor",
    "coord_hi_emg_peak_flexor",
    "coord_hi_emg_peak_ratio",
    "coord_hi_rms_extensor",
    "coord_hi_rms_flexor",
    "coord_hi_iemg_extensor",
    "coord_hi_iemg_flexor",
    "coord_hi_mean_emg_extensor",
    "coord_hi_mean_emg_flexor",
    "scalogram_hi_median_freq_extensor",
    "scalogram_hi_median_freq_flexor",
    "scalogram_hi_median_freq_ratio",
    "scalogram_hi_peak_freq_extensor",
    "scalogram_hi_peak_freq_flexor",
    "scalogram_hi_peak_freq_ratio",
    "scalogram_hi_time_to_peak_energy_extensor",
    "scalogram_hi_time_to_peak_energy_flexor",
    "band_power_hi_0-50Hz_extensor",
    "band_power_hi_50-100Hz_extensor",
    "band_power_hi_100-150Hz_extensor",
    "band_power_hi_150-200Hz_extensor",
    "band_power_hi_200-250Hz_extensor",
    "band_power_hi_250-300Hz_extensor",
    "band_power_hi_300-350Hz_extensor",
    "band_power_hi_350-400Hz_extensor",
    "band_power_hi_400-450Hz_extensor",
    "band_power_hi_450-500Hz_extensor",
    "band_power_hi_0-50Hz_flexor",
    "band_power_hi_50-100Hz_flexor",
    "band_power_hi_100-150Hz_flexor",
    "band_power_hi_150-200Hz_flexor",
    "band_power_hi_200-250Hz_flexor",
    "band_power_hi_250-300Hz_flexor",
    "band_power_hi_300-350Hz_flexor",
    "band_power_hi_350-400Hz_flexor",
    "band_power_hi_400-450Hz_flexor",
    "band_power_hi_450-500Hz_flexor",
    "hi_values_max",
    "hi_values_mean",
],

    "Osteopenia": [
        "Estimated bone mass kg"

    ],

    "Powerpenia": [
    "peak_power",
    "velocity_at_peak_power",
    "force_at_peak_power",
    "max_velocity",
    "max_force",
    "coord_hd_onset_diff_ms",
    "coord_hd_onset_diff_ms_abs",
    "coord_hd_cocontraction_ms",
    "coord_hd_cocontraction_normalized",
    "coord_hd_emg_peak_extensor",
    "coord_hd_emg_peak_flexor",
    "coord_hd_emg_peak_ratio",
    "coord_hd_envelope_peak_delay_ms",
    "coord_hd_envelope_peak_delay_ms_abs",
    "coord_hd_normalized_envelope_peak_delay",
    "coord_hd_normalized_envelope_peak_delay_abs",
    "coord_hd_peak_rfd",
    "scalogram_hd_median_freq_extensor",
    "scalogram_hd_median_freq_flexor",
    "scalogram_hd_median_freq_ratio",
    "scalogram_hd_peak_freq_extensor",
    "scalogram_hd_peak_freq_flexor",
    "scalogram_hd_peak_freq_ratio",
    "scalogram_hd_time_to_peak_energy_extensor",
    "scalogram_hd_time_to_peak_energy_flexor",
    "band_power_hd_0-50Hz_extensor",
    "band_power_hd_50-100Hz_extensor",
    "band_power_hd_100-150Hz_extensor",
    "band_power_hd_150-200Hz_extensor",
    "band_power_hd_200-250Hz_extensor",
    "band_power_hd_250-300Hz_extensor",
    "band_power_hd_300-350Hz_extensor",
    "band_power_hd_350-400Hz_extensor",
    "band_power_hd_400-450Hz_extensor",
    "band_power_hd_450-500Hz_extensor",
    "band_power_hd_0-50Hz_flexor",
    "band_power_hd_50-100Hz_flexor",
    "band_power_hd_100-150Hz_flexor",
    "band_power_hd_150-200Hz_flexor",
    "band_power_hd_200-250Hz_flexor",
    "band_power_hd_250-300Hz_flexor",
    "band_power_hd_300-350Hz_flexor",
    "band_power_hd_350-400Hz_flexor",
    "band_power_hd_400-450Hz_flexor",
    "band_power_hd_450-500Hz_flexor",
],

    "Sarcopenia": [
        "Global muscle kg",
        "Arm muscle (right) kg",
        "Arm muscle (left) kg",
        "Leg muscle (right) kg",
        "Leg muscle (left) kg",
        "Torso muscle kg",
        "ALMI"

    ]
}

RANDOM_SEED = 42

ACC_COLUMNS = ["x", "y", "z","gACCx", "gACCy", "gACCz"]
ACC_COLUMNS_HUB = ["x", "y", "z"]
ACC_COLUMNS_HD = ["gACCx", "gACCy", "gACCz"]
FORCE_COLUMN = "FORCE"
DISPLACEMENT_COLUMN = "DISPLACEMENT"
FS = 1000

"""
Functions to...

Available Functions
-------------------
[Public]
function1(...):

[Private]
_function2(...):
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #

# internal imports

# ------------------------------------------------------------------------------------------------------------------- #
# file specific constants
# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #
"""

Functions for performing cross-validation for model selection and hyperparameter tuning.

Available Functions
-------------------
[Public]
nested_cross_val(...): Performs nested cross-validation using GroupKFold to evaluate a machine learning model with hyperparameter tuning.
tune_production_model(...): Performs hyperparameter tuning using GroupKFold, X and y.
------------------
[Private]
None
------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from typing import Union, List, Dict, Any
from sklearn.base import RegressorMixin
from constants import RANDOM_SEED


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #
RANDOM_SEED = 42

EMG_Extensor_filt = "EMG_Extensor_filt"
EMG_Flexor_filt = "EMG_Flexor_filt"
EMG_Extensor = "EMG_Extensor"
EMG_Flexor = "EMG_Flexor"
EMG_CHANNELS = [EMG_Extensor, EMG_Flexor]