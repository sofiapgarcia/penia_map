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
        "rep_max_hi_onset_diff_ms",
        "rep_max_hi_onset_diff_ms_abs",
        "rep_max_hi_cocontraction_ms",
        "rep_max_hi_cocontraction_normalized",
        "rep_max_hi_emg_peak_extensor",
        "rep_max_hi_emg_peak_flexor",
        "rep_max_hi_emg_peak_ratio",
        "rep_max_hi_rms_extensor",
        "rep_max_hi_rms_flexor",
        "rep_max_hi_iemg_extensor",
        "rep_max_hi_iemg_flexor",
        "rep_max_hi_mean_emg_extensor",
        "rep_max_hi_mean_emg_flexor",
        "rep_max_hi_emg_overlap_area",
        "rep_max_hi_emg_overlap_area_normalized",
        "rep_max_hi_20-70Hz_extensor",
        "rep_max_hi_70-120Hz_extensor",
        "rep_max_hi_120-170Hz_extensor",
        "rep_max_hi_170-220Hz_extensor",
        "rep_max_hi_220-270Hz_extensor",
        "rep_max_hi_270-320Hz_extensor",
        "rep_max_hi_320-370Hz_extensor",
        "rep_max_hi_370-420Hz_extensor",
        "rep_max_hi_420-470Hz_extensor",
        "rep_max_hi_20-70Hz_flexor",
        "rep_max_hi_70-120Hz_flexor",
        "rep_max_hi_120-170Hz_flexor",
        "rep_max_hi_170-220Hz_flexor",
        "rep_max_hi_220-270Hz_flexor",
        "rep_max_hi_270-320Hz_flexor",
        "rep_max_hi_320-370Hz_flexor",
        "rep_max_hi_370-420Hz_flexor",
        "rep_max_hi_420-470Hz_flexor",
        "rep_max_hi_Median_Frequency_extensor",
        "rep_max_hi_Median_Frequency_flexor",
        "rep_max_hi_Peak_Frequency_extensor",
        "rep_max_hi_Peak_Frequency_flexor",
        "rep_max_hi_Median_Frequency_ratio",
        "rep_max_hi_Peak_Frequency_ratio",
        "mean_hi_onset_diff_ms",
        "mean_hi_onset_diff_ms_abs",
        "mean_hi_cocontraction_ms",
        "mean_hi_cocontraction_normalized",
        "mean_hi_emg_peak_extensor",
        "mean_hi_emg_peak_flexor",
        "mean_hi_emg_peak_ratio",
        "mean_hi_rms_extensor",
        "mean_hi_rms_flexor",
        "mean_hi_iemg_extensor",
        "mean_hi_iemg_flexor",
        "mean_hi_mean_emg_extensor",
        "mean_hi_mean_emg_flexor",
        "mean_hi_emg_overlap_area",
        "mean_hi_emg_overlap_area_normalized",
        "mean_hi_20-70Hz_extensor",
        "mean_hi_70-120Hz_extensor",
        "mean_hi_120-170Hz_extensor",
        "mean_hi_170-220Hz_extensor",
        "mean_hi_220-270Hz_extensor",
        "mean_hi_270-320Hz_extensor",
        "mean_hi_320-370Hz_extensor",
        "mean_hi_370-420Hz_extensor",
        "mean_hi_420-470Hz_extensor",
        "mean_hi_20-70Hz_flexor",
        "mean_hi_70-120Hz_flexor",
        "mean_hi_120-170Hz_flexor",
        "mean_hi_170-220Hz_flexor",
        "mean_hi_220-270Hz_flexor",
        "mean_hi_270-320Hz_flexor",
        "mean_hi_320-370Hz_flexor",
        "mean_hi_370-420Hz_flexor",
        "mean_hi_420-470Hz_flexor",
        "mean_hi_Median_Frequency_extensor",
        "mean_hi_Median_Frequency_flexor",
        "mean_hi_Peak_Frequency_extensor",
        "mean_hi_Peak_Frequency_flexor",
        "mean_hi_Median_Frequency_ratio",
        "mean_hi_Peak_Frequency_ratio",
        "hi_Median_Frequency_extensor_slope",
        "hi_Median_Frequency_flexor_slope",
        "hi_values_max",
        "hi_values_mean",
    ],
    "Osteopenia": [
        "Estimated bone mass kg",
        "Bone Mineral Density",
    ],
    "Powerpenia": [
        "rep_pp_hd_onset_diff_ms",
        "rep_pp_hd_onset_diff_ms_abs",
        "rep_pp_hd_cocontraction_ms",
        "rep_pp_hd_cocontraction_normalized",
        "rep_pp_hd_emg_peak_extensor",
        "rep_pp_hd_emg_peak_flexor",
        "rep_pp_hd_emg_peak_ratio",
        "rep_pp_hd_envelope_peak_delay_ms",
        "rep_pp_hd_envelope_peak_delay_ms_abs",
        "rep_pp_hd_normalized_envelope_peak_delay",
        "rep_pp_hd_normalized_envelope_peak_delay_abs",
        "rep_pp_hd_peak_rfd",
        "rep_pp_hd_emg_overlap_area",
        "rep_pp_hd_emg_overlap_area_normalized",
        "rep_pp_hd_20-70Hz_extensor",
        "rep_pp_hd_70-120Hz_extensor",
        "rep_pp_hd_120-170Hz_extensor",
        "rep_pp_hd_170-220Hz_extensor",
        "rep_pp_hd_220-270Hz_extensor",
        "rep_pp_hd_270-320Hz_extensor",
        "rep_pp_hd_320-370Hz_extensor",
        "rep_pp_hd_370-420Hz_extensor",
        "rep_pp_hd_420-470Hz_extensor",
        "rep_pp_hd_20-70Hz_flexor",
        "rep_pp_hd_70-120Hz_flexor",
        "rep_pp_hd_120-170Hz_flexor",
        "rep_pp_hd_170-220Hz_flexor",
        "rep_pp_hd_220-270Hz_flexor",
        "rep_pp_hd_270-320Hz_flexor",
        "rep_pp_hd_320-370Hz_flexor",
        "rep_pp_hd_370-420Hz_flexor",
        "rep_pp_hd_420-470Hz_flexor",
        "rep_pp_hd_Median_Frequency_extensor",
        "rep_pp_hd_Median_Frequency_flexor",
        "rep_pp_hd_Peak_Frequency_extensor",
        "rep_pp_hd_Peak_Frequency_flexor",
        "rep_pp_hd_Median_Frequency_ratio",
        "rep_pp_hd_Peak_Frequency_ratio",
        "mean_hd_onset_diff_ms",
        "mean_hd_onset_diff_ms_abs",
        "mean_hd_cocontraction_ms",
        "mean_hd_cocontraction_normalized",
        "mean_hd_emg_peak_extensor",
        "mean_hd_emg_peak_flexor",
        "mean_hd_emg_peak_ratio",
        "mean_hd_envelope_peak_delay_ms",
        "mean_hd_envelope_peak_delay_ms_abs",
        "mean_hd_normalized_envelope_peak_delay",
        "mean_hd_normalized_envelope_peak_delay_abs",
        "mean_hd_peak_rfd",
        "mean_hd_emg_overlap_area",
        "mean_hd_emg_overlap_area_normalized",
        "mean_hd_20-70Hz_extensor",
        "mean_hd_70-120Hz_extensor",
        "mean_hd_120-170Hz_extensor",
        "mean_hd_170-220Hz_extensor",
        "mean_hd_220-270Hz_extensor",
        "mean_hd_270-320Hz_extensor",
        "mean_hd_320-370Hz_extensor",
        "mean_hd_370-420Hz_extensor",
        "mean_hd_420-470Hz_extensor",
        "mean_hd_20-70Hz_flexor",
        "mean_hd_70-120Hz_flexor",
        "mean_hd_120-170Hz_flexor",
        "mean_hd_170-220Hz_flexor",
        "mean_hd_220-270Hz_flexor",
        "mean_hd_270-320Hz_flexor",
        "mean_hd_320-370Hz_flexor",
        "mean_hd_370-420Hz_flexor",
        "mean_hd_420-470Hz_flexor",
        "mean_hd_Median_Frequency_extensor",
        "mean_hd_Median_Frequency_flexor",
        "mean_hd_Peak_Frequency_extensor",
        "mean_hd_Peak_Frequency_flexor",
        "mean_hd_Median_Frequency_ratio",
        "mean_hd_Peak_Frequency_ratio",
        "hd_Median_Frequency_extensor_slope",
        "hd_Median_Frequency_flexor_slope",
        "peak_power",
        "velocity_at_peak_power",
        "force_at_peak_power",
        "max_velocity",
        "max_force",
        "peak_power_mass",
        "peak_power_height_squared"
    ],
    "Sarcopenia": [
        "Global muscle kg",
        "Arm muscle (right) kg",
        "Arm muscle (left) kg",
        "Leg muscle (right) kg",
        "Leg muscle (left) kg",
        "ALMI",
    ],
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


RANDOM_SEED = 42

EMG_Extensor_filt = "EMG_Extensor_filt"
EMG_Flexor_filt = "EMG_Flexor_filt"
EMG_Extensor = "EMG_Extensor"
EMG_Flexor = "EMG_Flexor"
EMG_CHANNELS = [EMG_Extensor, EMG_Flexor]

#Antes
best_models_penia_map = {
    "Estimated metabolic age": [
        "rep_pp_hd_cocontraction_ms",
        "peak_power_mass",
        "ALMI"
    ],
    "peak_power": [
        "hi_values_max",
        "Arm muscle (right) kg"
    ],
    "Bone Mineral Density": [
        "ALMI",
        "Gender"
    ],
    "Age (y)": ["rep_pp_hd_cocontraction_ms",
                "Bone Mineral Density"
    ]
}


best_models_all_variables = {
    "Estimated metabolic age": [
        "Body mass index (BMI)",
        "rep_pp_hd_cocontraction_ms"
    ],
    "peak_power": [
        "hi_values_max",
        "Arm muscle (right) kg"
    ],
    "Bone Mineral Density": [
        "Global muscle kg",
        "ALMI",
        "Body mass(kg)"
    ],
    "Age (y)": ["Body mass index (BMI)",
                "rep_pp_hd_Median_Frequency_flexor"
    ]
}
"""
best_models_all_variables = {
    "Estimated metabolic age": ['rep_pp_hd_Peak_Frequency_extensor', 'rep_max_hi_Median_Frequency_flexor', 'peak_power_mass', 'Leg muscle (right) kg', 'Gender', 'Body mass index (BMI)'],
    "peak_power": ["rep_pp_hd_onset_diff_ms", "rep_pp_hd_normalized_envelope_peak_delay", "rep_max_hi_Peak_Frequency_extensor", "hi_values_max", "Gender", "Body mass(kg)"],
    "Bone Mineral Density": ['rep_pp_hd_emg_peak_extensor', 'rep_max_hi_emg_overlap_area', 'Global muscle kg', 'ALMI'],
    "Age (y)": ['rep_max_hi_Median_Frequency_flexor', 'rep_max_hi_Peak_Frequency_flexor', 'peak_power_mass', 'Leg muscle (right) kg', 'Gender', 'Body mass index (BMI)']
}
best_models_penia_map = {
    "Estimated metabolic age": ['rep_pp_hd_emg_overlap_area', 'rep_max_hi_emg_peak_ratio', 'hi_Median_Frequency_extensor_slope', 'Arm muscle (left) kg','Leg muscle (right) kg', 'ALMI'],
    "peak_power": ['rep_pp_hd_onset_diff_ms_abs', 'rep_pp_hd_emg_peak_ratio', 'rep_pp_hd_normalized_envelope_peak_delay_abs', 'rep_max_hi_mean_emg_extensor', 'rep_max_hi_Peak_Frequency_extensor', 'Arm muscle (right) kg'],
    "Bone Mineral Density": ['rep_pp_hd_cocontraction_ms', 'rep_max_hi_cocontraction_ms', 'Global muscle kg', 'ALMI'],
    "Age (y)": ['rep_pp_hd_emg_peak_ratio', 'rep_max_hi_onset_diff_ms', 'hd_Median_Frequency_flexor_slope', 'Arm muscle (left) kg', 'Leg muscle (right) kg', 'ALMI']
}

best_models_all_variables = {
    "Estimated metabolic age": ['max_velocity', 'Gender', 'Body mass index (BMI)'],
    "peak_power": ['rep_pp_hd_emg_peak_flexor', 'rep_max_hi_emg_overlap_area', 'hi_values_max', 'Height(cm)', 'Gender'],
    "Bone Mineral Density": ['Global muscle kg', 'ALMI'],
    "Age (y)": ['Arm muscle (left) kg', 'Height(cm)']
}
best_models_penia_map = {
    "Estimated metabolic age": ['max_velocity', 'rep_pp_hd_cocontraction_ms', 'rep_pp_hd_emg_peak_ratio', 'rep_pp_hd_Peak_Frequency_extensor', 'rep_max_hi_Median_Frequency_flexor', 'ALMI'],
    "peak_power": ['hi_values_max', 'Arm muscle (right) kg'],
    "Bone Mineral Density": ['Global muscle kg', 'ALMI'],
    "Age (y)": ['rep_pp_hd_cocontraction_ms']
}
"""
