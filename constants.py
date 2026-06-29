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
        "Hand Grip_F",
        "1RM_BP_F",
        "1RM_SQ_F"
    ],

    "Osteopenia": [
        "BMC_pernas",
        "BMC_perna_direita",
        "BMC_perna_esquerda",
        "BMC_dif_pernas",
        "BMC_tronco",
        "BMC_bracos",
        "BMC_braco_direito",
        "BMC_braco_esquerdo",
        "BMC_dif_bracos",
        "BMC_androide",
        "BMC_ginoide",
        "BMC_total",
        "BMD",
        "BMD_cabeca",
        "BMD_bracos",
        "BMD_pernas",
        "BMD_tronco",
        "BMD_costelas",
        "BMD_coluna",
        "BMD_pelve",
        "T_score",
        "Z_score",
        "Centil"

    ],

    "Powerpenia": [
        "Peak_Power_BP_F",
        "Peak_Power_SQ_F"
    ],

    "Sarcopenia": [
        "Magro_total",
        "Magro_pernas",
        "Magro_tronco"
        "Massa_magra_bracos",
        "Massa_magra_braco_direito",
        "Massa_magra_braco_esquerdo",
        "Massa_magra_dif_bracos",
        "Massa_magra_perna_direita",
        "Massa_magra_perna_esquerda",
        "Massa_magra_dif_pernas",
        "Massa_magra_androide",
        "Massa_magra_ginoide",
        "Massa_magra_tronco_direito",
        "Massa_magra_tronco_esquerdo",
        "Massa_magra_tronco_dif",
        "Massa_magra_direita",
        "Massa_magra_esquerda",
        "Massa_magra_dif",
        "Perc_tecido_magro",
        "RSMI"
    ],

    "Body Fat": [
        "Perc_gordura_total",
        "Perc_gordura_pernas",
        "Perc_gordura_tronco",
        "Perc_gordura_androide",
        "Perc_gordura_ginoide",
        "Perc_gordura_bracos_total",
        "Perc_gordura_braco_direito",
        "Perc_gordura_braco_esquerdo",
        "Perc_gordura_dif_bracos",
        "Perc_gordura_perna_direita",
        "Perc_gordura_perna_esquerda",
        "Perc_gordura_dif_pernas",
        "Dist_gordura_bracos",
        "Dist_gordura_pernas",
        "Dist_gordura_tronco",
        "Dist_gordura_androide",
        "Dist_gordura_ginoide",
        "Perc_gordura_regiao_total",
        "Gordo_total",
        "Massa_gordura_bracos",
        "Massa_gordura_braco_direito",
        "Massa_gordura_braco_esquerdo",
        "Massa_gordura_dif_bracos",
        "Massa_gordura_pernas",
        "Massa_gordura_perna_direita",
        "Massa_gordura_perna_esquerda",
        "Massa_gordura_dif_pernas",
        "Massa_gordura_tronco",
        "Massa_gordura_androide",
        "Massa_gordura_ginoide",
        "VAT_volume",
        "VAT_massa_gordura",
        "VAT_area",
        "TAS_volume",
        "TAS_massa_gordura",
        "TAS_area",
        "Taxa_A_G"
    ],

    "Metabolic": [
        "RMR",
        "Daily calorie intake (DCI)"
    ],

    "Anthropometric": [
        "BMI_DXA",
        "Altura",
        "Massa Corporal"
    ],

    "Demographics": [
        "Idade",
        "Gender_M",
        "IPAQ_Low",
        "IPAQ_Moderate",
        "HHIE-S (Classification)_No Handicap",
        "Marital Status_Married",
        "Marital Status_Single",
        "Marital Status_União de Facto",
        "Marital Status_Widow",
        "Education Level_Ensino Superior",
        "Education Level_Pós-Graduação",
        "Education Level_Secundário",
        "MET/Week",
    ],

    "Others": [
        "Tecido"
    ]
}

RANDOM_SEED = 42

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