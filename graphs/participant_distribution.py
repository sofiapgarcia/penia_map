"""
Participantes - histograma de idades (todos juntos) + gráfico circular do sexo
Mesma estética do Penia Map (roxo, grelha leve, texto cinzento, fundo branco).

Gera (na pasta OUTPUT_DIR):
    participantes_idades.png / .svg   (histograma)
    participantes_sexo.png / .svg     (gráfico circular / donut)

Uso:
    python graph_participantes.py                 # usa EXCEL_PATH definido abaixo
    python graph_participantes.py dados.xlsx      # ou passa o ficheiro na linha de comandos

Dependências:
    pip install pandas numpy matplotlib openpyxl
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

# =====================================================================
# 1) CONFIGURAÇÃO
# =====================================================================

EXCEL_PATH = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\report_participants_valids.xlsx"
SHEET_NAME = 0

# --- participantes a excluir (iguais aos do script do Penia Map) ---
ID_COL = "ID"
IDS_TO_REMOVE = [4, 14, 17,22,24]

# --- colunas ---
AGE_COL = "Age (y)"   # idade em anos (numérica)
SEX_COL = "Gender"
FEMALE_VALUE = 0
MALE_VALUE = 1

# --- histograma ---
BIN_WIDTH = 5                 # largura de cada barra, em anos
HIGHLIGHT_GROUPS = False       # destaca as barras dos 3 grupos usados no Penia Map
GROUP_RANGES = [              # (nome, idade mínima, idade máxima EXCLUÍDA)
    ("Jovens (18-24)", 18, 25),
    ("Meia-idade (45-64)", 45, 65),
    ("Idosos (≥65)", 65, 200),
]
SHOW_COUNTS_ON_BARS = True    # escreve o n por cima de cada barra
X_LABEL = "Idade (anos)"
Y_LABEL = "N.º de participantes"

# --- gráfico circular ---
LABEL_FEMALE = "Mulheres"
LABEL_MALE = "Homens"
CENTER_LABEL = "participantes"

# --- aparência (mesmas cores do Penia Map) ---
COLOR_MAIN = "#1d78ba"
COLOR_LIGHT = "#c9eaee"
COLOR_LIGHT_TEXT = "#2B2566"
TEXT_GRAY = "#6B6B6B"
GRID_COLOR = "#E3E3E3"
OUTPUT_PREFIX = "participantes"
OUTPUT_DIR = r"C:\Users\Sofia\Desktop\Tese\apresentação\graficos"

# =====================================================================
# 2) DADOS
# =====================================================================


def _check_config():
    missing = [
        name
        for name, value in globals().items()
        if name.isupper() and isinstance(value, str) and value.startswith("PREENCHER")
    ]
    if missing:
        sys.exit("Falta preencher no topo do script:\n  - " + "\n  - ".join(sorted(missing)))


def _norm_value(v):
    """1, 1.0 e "1" ficam iguais; ignora maiúsculas e espaços."""
    if isinstance(v, (int, float, np.integer, np.floating)) and not pd.isna(v) and float(v).is_integer():
        v = int(v)
    return str(v).strip().lower()


def _to_number(series):
    if series.dtype == object:
        series = series.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(series, errors="coerce")


def load_data(excel_path):
    df = pd.read_excel(excel_path, sheet_name=SHEET_NAME)
    df.columns = [str(c).strip() for c in df.columns]

    needed = [ID_COL, AGE_COL, SEX_COL]
    absent = [c for c in needed if c not in df.columns]
    if absent:
        sys.exit(
            f"Colunas não encontradas no Excel: {absent}\n"
            f"Colunas disponíveis: {list(df.columns)}"
        )

    ids_norm = df[ID_COL].map(_norm_value)
    to_remove = {_norm_value(i) for i in IDS_TO_REMOVE}
    mask = ids_norm.isin(to_remove)
    print(f"Participantes no ficheiro: {len(df)}")
    print(f"Removidos ({int(mask.sum())}): {sorted(df.loc[mask, ID_COL].astype(str).str.strip())}")
    not_found = [i for i in IDS_TO_REMOVE if _norm_value(i) not in set(ids_norm)]
    if not_found:
        print(f"ATENÇÃO: IDs a remover que não existem no ficheiro: {not_found}")
    df = df[~mask].copy()
    print(f"Participantes depois da remoção: {len(df)}")

    df[AGE_COL] = _to_number(df[AGE_COL])
    n_sem_idade = int(df[AGE_COL].isna().sum())
    if n_sem_idade:
        print(f"ATENÇÃO: {n_sem_idade} participante(s) sem idade válida (ignorados no histograma).")
    return df


# =====================================================================
# 3) ESTILO E GRÁFICOS
# =====================================================================


def _setup_font():
    preferred = ["Inter", "DM Sans", "Helvetica Neue", "Helvetica", "Arial"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = next((n for n in preferred if n in available), "DejaVu Sans")


def _in_group(center):
    return any(lo <= center < hi for _, lo, hi in GROUP_RANGES)


def draw_age_histogram(ages):
    ages = ages.dropna().to_numpy()
    n = len(ages)
    lo = int(np.floor(ages.min() / BIN_WIDTH) * BIN_WIDTH)
    hi = int(np.ceil((ages.max() + 1e-9) / BIN_WIDTH) * BIN_WIDTH)
    if hi <= ages.max():
        hi += BIN_WIDTH
    edges = np.arange(lo, hi + BIN_WIDTH, BIN_WIDTH)
    counts, _ = np.histogram(ages, bins=edges)
    lefts = edges[:-1]
    centers = lefts + BIN_WIDTH / 2

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = [
        COLOR_MAIN if (not HIGHLIGHT_GROUPS or _in_group(c)) else COLOR_LIGHT
        for c in centers
    ]
    ax.bar(lefts, counts, width=BIN_WIDTH, align="edge", color=colors,
           edgecolor="white", linewidth=2.5, zorder=3)

    if SHOW_COUNTS_ON_BARS:
        for x, c in zip(centers, counts):
            if c > 0:
                ax.text(x, c + max(counts) * 0.025, str(c), ha="center", va="bottom",
                        fontsize=10.5, fontweight="bold", color="#111111", zorder=4)

    ax.set_ylim(0, max(counts) * 1.2)
    ax.set_xlim(edges[0] - 0.2, edges[-1] + 0.2)
    ax.set_xticks(centers)
    ax.set_xticklabels([f"{int(a)}–{int(a + BIN_WIDTH - 1)}" for a in lefts],
                       fontsize=9.5, color=TEXT_GRAY)
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.tick_params(axis="y", labelsize=9.5, labelcolor=TEXT_GRAY, length=0)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#BDBDBD")
    ax.set_xlabel(X_LABEL, fontsize=10.5, color=TEXT_GRAY, labelpad=10)
    ax.set_ylabel(Y_LABEL, fontsize=10.5, color=TEXT_GRAY, labelpad=10)

    ax.set_title(
        f"n = {n}   ·   idade média = {ages.mean():.1f} ± {ages.std(ddof=1):.1f} anos"
        f"   ·   intervalo {ages.min():.0f}–{ages.max():.0f} anos",
        loc="left", fontsize=10.5, color=TEXT_GRAY, pad=14)

    if HIGHLIGHT_GROUPS:
        handles = [
            Patch(facecolor=COLOR_MAIN, label="Grupos usados na comparação do Penia Map"),
            Patch(facecolor=COLOR_LIGHT, label="Restantes participantes"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
                   fontsize=10, bbox_to_anchor=(0.5, -0.01))
        fig.subplots_adjust(bottom=0.24)
    fig.tight_layout(rect=(0, 0.06 if HIGHLIGHT_GROUPS else 0, 1, 1))
    return fig


def draw_sex_donut(n_female, n_male):
    total = n_female + n_male
    values = [n_female, n_male]
    pcts = [v / total * 100 for v in values]
    colors = [COLOR_MAIN, COLOR_LIGHT]
    text_colors = ["white", COLOR_LIGHT_TEXT]
    names = [LABEL_FEMALE, LABEL_MALE]

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    width = 0.42
    wedges, _ = ax.pie(values, colors=colors, startangle=90, counterclock=False,
                       wedgeprops=dict(width=width, edgecolor="white", linewidth=3))

    for w, v, tc in zip(wedges, values, text_colors):
        ang = np.deg2rad((w.theta1 + w.theta2) / 2)
        r = 1 - width / 2
        ax.text(r * np.cos(ang), r * np.sin(ang), f"{v}", ha="center", va="center",
                fontsize=18, fontweight="bold", color=tc)

    ax.text(0, 0.08, f"{total}", ha="center", va="center", fontsize=30,
            fontweight="bold", color="#111111")
    ax.text(0, -0.20, CENTER_LABEL, ha="center", va="center", fontsize=11, color=TEXT_GRAY)
    ax.set_aspect("equal")

    handles = [Patch(facecolor=c, label=f"{nm} · {pct:.0f}%")
               for c, nm, pct in zip(colors, names, pcts)]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=11.5, bbox_to_anchor=(0.5, 0.02))
    fig.subplots_adjust(top=0.98, bottom=0.12)
    return fig


# =====================================================================
# 4) EXECUÇÃO
# =====================================================================


def _save(fig, name):
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        out = out_dir / f"{OUTPUT_PREFIX}_{name}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Guardado: {out.resolve()}")
    plt.close(fig)


def main():
    excel = sys.argv[1] if len(sys.argv) > 1 else EXCEL_PATH
    if len(sys.argv) > 1:
        globals()["EXCEL_PATH"] = excel
    _check_config()
    _setup_font()

    df = load_data(excel)

    ages = df[AGE_COL].dropna()
    print(f"\nIdade: n = {len(ages)}, média = {ages.mean():.1f} ± {ages.std(ddof=1):.1f}, "
          f"intervalo = {ages.min():.0f}–{ages.max():.0f}")

    sex_norm = df[SEX_COL].map(_norm_value)
    n_f = int((sex_norm == _norm_value(FEMALE_VALUE)).sum())
    n_m = int((sex_norm == _norm_value(MALE_VALUE)).sum())
    print(f"Sexo: {LABEL_FEMALE} = {n_f}, {LABEL_MALE} = {n_m}")
    if n_f + n_m != len(df):
        print(f"ATENÇÃO: {len(df) - n_f - n_m} participante(s) sem sexo definido "
              f"(nem {FEMALE_VALUE} nem {MALE_VALUE}).")
    print()

    _save(draw_age_histogram(df[AGE_COL]), "idades")
    _save(draw_sex_donut(n_f, n_m), "sexo")


if __name__ == "__main__":
    main()