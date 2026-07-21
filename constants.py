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
        "H1",
        "H2",
        "H3",
        "HI_max"
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
        "peak_power_mass",
        "peak_power_height_squared"

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


def nested_cross_val(X: pd.DataFrame, y: pd.Series, estimator: RegressorMixin,
                    splits_outer: int, splits_inner: int,
                     param_grid: Union[List[Dict[str, Any]], Dict[str, Any]]
                     ) -> pd.DataFrame:
    """
    Performs nested cross-validation using KFold to evaluate a machine learning regression model with hyperparameter tuning.

    The function applies a nested cross-validation strategy, where the outer loop performs model evaluation
    and the inner loop performs hyperparameter tuning using GridSearchCV.

    The function generates a pandas.DataFrame containing the hyperparameters of the model as well as:
    - avg. R² score of the inner GridSearch
    - avg. MAE of the inner GridSearch
    - R² score per outer fold
    - MAE per outer fold
    - mean and std of R² over all folds
    - mean and std of MAE over all folds

    :param X: pandas.DataFrame containing the feature matrix
    :param y: pandas.Series containing the target values
    :param estimator: the estimator to be evaluated
    :param param_grid: the parameter grid for the estimator
    :param splits_outer: number of outer cross-validation splits.
    :param splits_inner: number of inner cross-validation splits.
    :return: pandas.DataFrame containing the best hyperparameters found for each outer fold,
        inner cross-validation R² and MAE scores, outer fold R² and MAE scores, and the overall
        mean and standard deviation of R² and MAE across all outer folds.
    """

    # lists for holding the scores of the outer folds and the info of each fold
    fold_r2_scores = []
    fold_mae_scores = []
    fold_info = []

    # set up cross-validation for outer loop
    outer_cv = KFold(n_splits=splits_outer, shuffle=True, random_state=RANDOM_SEED)

    # set up inner cross-validation for hyperparameter tuning using gridSearch
    inner_cv = KFold(n_splits=splits_inner, shuffle=True, random_state=RANDOM_SEED)
    grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring='r2',
                               n_jobs=-1, cv=inner_cv, verbose=1, refit=True)

    # run outer loop
    for fold_num, (train_idx, valid_idx) in enumerate(outer_cv.split(X, y), start=1):

        # get the corresponding X, y
        X_train = X.iloc[train_idx]
        X_val = X.iloc[valid_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[valid_idx]

        # run inner loop for hyperparameter optimization
        grid_search.fit(X_train, y_train)

        # test the best estimator on the validation set of the current fold (this is possible as refit=True)
        y_pred = grid_search.best_estimator_.predict(X_val)

        # Calculate metrics for outer fold
        fold_r2 = r2_score(y_val, y_pred)
        fold_mae = mean_absolute_error(y_val, y_pred)

        # Get metrics from inner fold
        inner_r2 = grid_search.best_score_

        # Calculate inner MAE by doing cross-validation manually
        inner_mae_scores = []
        for train_idx_inner, val_idx_inner in inner_cv.split(X_train, y_train):
            X_train_inner = X_train.iloc[train_idx_inner]
            X_val_inner = X_train.iloc[val_idx_inner]
            y_train_inner = y_train.iloc[train_idx_inner]
            y_val_inner = y_train.iloc[val_idx_inner]

            # Fit with best params and calculate MAE
            grid_search.best_estimator_.fit(X_train_inner, y_train_inner)
            y_pred_inner = grid_search.best_estimator_.predict(X_val_inner)
            mae_inner = mean_absolute_error(y_val_inner, y_pred_inner)
            inner_mae_scores.append(mae_inner)

        inner_mae = np.mean(inner_mae_scores)
        best_params = grid_search.best_params_

        # print the results
        print(f'\nResults of fold {fold_num}/{splits_outer}')
        print(f'Inner fold R² score (avg.): {inner_r2:.4f}')
        print(f'Inner fold MAE (avg.):      {inner_mae:.4f}')
        print(f'Best params: {best_params}')
        print(f'--> Outer fold R² score: {fold_r2:.4f}')
        print(f'--> Outer fold MAE:      {fold_mae:.4f}')
        print('-------------------------')

        # append fold scores
        fold_r2_scores.append(fold_r2)
        fold_mae_scores.append(fold_mae)

        # append the info for the fold
        best_params.update({
            'inner_r2': inner_r2,
            'inner_mae': inner_mae,
            'fold_r2': fold_r2,
            'fold_mae': fold_mae
        })
        fold_info.append(best_params)

    # calculate the average R² and MAE scores over all outer folds
    outer_fold_avg_r2 = np.mean(fold_r2_scores)
    outer_fold_std_r2 = np.std(fold_r2_scores)

    outer_fold_avg_mae = np.mean(fold_mae_scores)
    outer_fold_std_mae = np.std(fold_mae_scores)

    # print the results
    print(f'\nAverage R² score over all folds:  {outer_fold_avg_r2:.4f} +/- {outer_fold_std_r2:.4f}')
    print(f'Average MAE over all folds:      {outer_fold_avg_mae:.4f} +/- {outer_fold_std_mae:.4f}')
    print('### ----------------------------------------- ###')

    # generate a pandas DataFrame containing the info
    info_df = pd.DataFrame(fold_info)

    # add the average scores
    # (adding here to not have it to re-calculate again. The entire column thus has the same value)
    info_df['estimator_avg_r2'] = outer_fold_avg_r2
    info_df['estimator_std_r2'] = outer_fold_std_r2

    info_df['estimator_avg_mae'] = outer_fold_avg_mae
    info_df['estimator_std_mae'] = outer_fold_std_mae

    return info_df