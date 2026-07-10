import os
import pandas as pd
import matplotlib.pyplot as plt
import pingouin as pg


def plot_bland_altman_pairs(
    df: pd.DataFrame,
    variable_mapping: dict,
    save_dir: str = None,
    show_plot: bool = True
) -> None:
    """
    Generate one Bland-Altman plot at a time for each DXA-TANITA variable pair
    defined in variable_mapping.

    :param df: DataFrame containing both DXA and TANITA columns, one row per subject
    :param variable_mapping: dict mapping DXA column name -> TANITA column name
    :param save_dir: if provided, path to directory where each figure will be saved as PNG
    :param show_plot: if True, displays each figure with plt.show() before moving to the next
    :return: None
    """
    valid_pairs = _filter_valid_pairs(df, variable_mapping)

    if not valid_pairs:
        print("Nenhum par válido encontrado para plotar.")
        return

    for dxa_col, tanita_col in valid_pairs:
        fig, ax = plt.subplots(figsize=(7, 6))
        _plot_single_bland_altman(df, dxa_col, tanita_col, ax)
        fig.tight_layout()

        if save_dir:
            _save_figure(fig, save_dir, dxa_col, tanita_col)

        if show_plot:
            plt.show()
        else:
            plt.close(fig)


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
            print(f"Aviso: par ('{dxa_col}', '{tanita_col}') ignorado - coluna em falta.")
            continue

        n = df[[dxa_col, tanita_col]].dropna().shape[0]
        if n < 3:
            print(f"Aviso: par ('{dxa_col}', '{tanita_col}') ignorado - apenas {n} observações válidas.")
            continue

        valid.append((dxa_col, tanita_col))

    return valid


def _plot_single_bland_altman(df: pd.DataFrame, dxa_col: str, tanita_col: str, ax: plt.Axes) -> None:
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

    ax.set_title(f"{dxa_col} vs {tanita_col}", fontsize=11)
    ax.set_xlabel("Média (DXA, TANITA)")
    ax.set_ylabel("Diferença (DXA - TANITA)")


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
    print(f"Figura guardada em: {filepath}")