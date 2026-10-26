"""
Functions to generate Bland-Altman plots for DXA-TANITA comparisons.

Available Functions
-------------------
[Public]
plot_bland_altman_pairs(...): Generate Bland-Altman plots for variable pairs.

[Private]
_filter_valid_pairs(...): Filter valid DXA-TANITA variable pairs.
_plot_single_bland_altman(...): Generate a single Bland-Altman plot.
_save_figure(...): Save a Bland-Altman plot as a PNG file.
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import os
import pandas as pd
import matplotlib.pyplot as plt
import pingouin as pg

# internal imports
from reliability_analysis_tanita.ICC import DXA_VARS_IN_GRAMS , _convert_dxa_grams_to_kg

# ------------------------------------------------------------------------------------------------------------------- #
# file specific constants
# ------------------------------------------------------------------------------------------------------------------- #
DXA_DISPLAY_NAMES = {
    "Massa Corporal": "Body mass",
    "BMI_DXA": "Body Mass Index",
    "Perc_gordura_total": "Total fat (%)",
    "Perc_gordura_braco_direito": "Right arm fat (%)",
    "Perc_gordura_braco_esquerdo": "Left arm fat (%)",
    "Perc_gordura_perna_direita": "Right leg fat (%)",
    "Perc_gordura_perna_esquerda": "Left leg fat (%)",
    "Perc_gordura_tronco": "Torso fat (%)",
    "Magro_total": "Global muscle mass",
    "Massa_magra_braco_direito": "Right arm muscle mass",
    "Massa_magra_braco_esquerdo": "Left arm muscle mass",
    "Massa_magra_perna_direita": "Right leg muscle mass",
    "Massa_magra_perna_esquerda": "Left leg muscle mass",
    "Magro_tronco": "Torso muscle mass",
    "BMC_total": "Bone mineral content",
    "Altura": "Height",
}

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def plot_bland_altman_pairs(
    df: pd.DataFrame,
    variable_mapping: dict,
    save_dir: str = None,
    show_plot: bool = True
) -> None:
    """
    Generate Bland-Altman plots for each DXA-TANITA variable pair defined in
    variable_mapping.

    :param df: DataFrame containing both DXA and TANITA columns, one row per subject
    :param variable_mapping: dict mapping DXA column name -> TANITA column name
    :param save_dir: if provided, path to directory where each figure will be saved as PNG
    :param show_plot: if True, displays all figures at once with a single plt.show() call
    :return: None
    """
    df = _convert_dxa_grams_to_kg(df, variable_mapping)
    valid_pairs = _filter_valid_pairs(df, variable_mapping)

    if not valid_pairs:
        print("No valid pairs found to plot.")
        return

    figs = []
    for dxa_col, tanita_col in valid_pairs:
        fig, ax = plt.subplots(figsize=(7, 6))
        _plot_single_bland_altman(df, dxa_col, tanita_col, ax)
        fig.tight_layout()

        if save_dir:
            _save_figure(fig, save_dir, dxa_col, tanita_col)

        figs.append(fig)

    if show_plot:
        plt.show()
    else:
        for fig in figs:
            plt.close(fig)

# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _filter_valid_pairs(df: pd.DataFrame, variable_mapping: dict) -> list:
    """
    Filter variable_mapping to only pairs where both columns exist in df and have
    at least 3 non-missing paired observations.

    :param df: source DataFrame
    :param variable_mapping: dict mapping DXA column name -> TANITA column name
    :return: list of (dxa_col, tanita_col) tuples that are safe to plot
    """
    valid = []
    for dxa_col, tanita_col in variable_mapping.items():
        if dxa_col not in df.columns or tanita_col not in df.columns:
            print(f"Warning: pair ('{dxa_col}', '{tanita_col}') skipped - missing column.")
            continue

        n = df[[dxa_col, tanita_col]].dropna().shape[0]
        if n < 3:
            print(f"Warning: pair ('{dxa_col}', '{tanita_col}') skipped - only {n} valid observations.")
            continue

        valid.append((dxa_col, tanita_col))

    return valid


def _plot_single_bland_altman(df: pd.DataFrame, dxa_col: str, tanita_col: str, ax: plt.Axes, dxa_display_names=DXA_DISPLAY_NAMES) -> None:
    """
    Draw a single Bland-Altman plot for one DXA-TANITA pair on a given matplotlib axis.

    :param df: source DataFrame
    :param dxa_col: DXA variable name
    :param tanita_col: TANITA variable name
    :param ax: matplotlib Axes object to draw on
    :return: None
    """
    pair_df = df[[dxa_col, tanita_col]].dropna()

    pg.plot_blandaltman(
        pair_df[dxa_col],
        pair_df[tanita_col],
        ax=ax,
        annotate=True
    )

    ax.set_title(dxa_display_names.get(dxa_col, dxa_col), fontsize=11)
    ax.set_xlabel("Mean (DXA, Bioimpedance scale)")
    ax.set_ylabel("Difference (DXA - Bioimpedance scale)")


def _save_figure(fig: plt.Figure, save_dir: str, dxa_col: str, tanita_col: str) -> None:
    """
    Save the given figure to disk as a PNG file inside save_dir, named after the variable pair.

    :param fig: matplotlib Figure to save
    :param save_dir: destination directory path
    :param dxa_col: DXA variable name, used to build the filename
    :param tanita_col: TANITA variable name, used to build the filename
    :return: None
    """
    os.makedirs(save_dir, exist_ok=True)
    safe_name = f"{dxa_col}_vs_{tanita_col}".replace(" ", "_").replace("/", "-")
    filepath = os.path.join(save_dir, f"bland_altman_{safe_name}.png")
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {filepath}")