"""
Functions to generate demographic plots of the study participants (age histogram and sex donut chart).

Available Functions
-------------------
[Public]
load_data(...): Load the participants Excel file, remove excluded IDs and convert age to numeric.
draw_age_histogram(...): Draw the age histogram of all participants.
draw_sex_donut(...): Draw the donut chart with the sex distribution.
main(...): Run the full pipeline (load data, print statistics, save both figures).

[Private]
_check_config(...): Check that no configuration value was left unfilled.
_to_number(...): Convert a Series to numeric, accepting decimal commas.
_setup_font(...): Set the matplotlib font to the first preferred font available.
_save(...): Save a figure as PNG and SVG in the output folder.
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

# ------------------------------------------------------------------------------------------------------------------- #
# CONFIGURATION
# ------------------------------------------------------------------------------------------------------------------- #

EXCEL_PATH = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\report_participants_valids.xlsx"
SHEET_NAME = 0

# --- participants to exclude ---
ID_COL = "ID"
IDS_TO_REMOVE = [4, 14, 17, 22, 24]

# --- columns ---
AGE_COL = "Age (y)"   # age in years (numeric)
SEX_COL = "Gender"
FEMALE_VALUE = 0
MALE_VALUE = 1

# --- histogram ---
BIN_WIDTH = 5                  # width of each bar, in years
GROUP_RANGES = [
    ("Jovens (18-24)", 18, 25),
    ("Meia-idade (45-64)", 45, 65),
    ("Idosos (≥65)", 65, 200),
]
SHOW_COUNTS_ON_BARS = True     # write the n above each bar
X_LABEL = "Idade (anos)"
Y_LABEL = "N.º de participantes"

# --- donut chart ---
LABEL_FEMALE = "Mulheres"
LABEL_MALE = "Homens"
CENTER_LABEL = "participantes"

# --- appearance ---
COLOR_MAIN = "#1d78ba"
COLOR_LIGHT = "#c9eaee"
COLOR_LIGHT_TEXT = "#2B2566"
TEXT_GRAY = "#6B6B6B"
GRID_COLOR = "#E3E3E3"
OUTPUT_PREFIX = "participantes"
OUTPUT_DIR = r"C:\Users\Sofia\Desktop\Tese\apresentação\graficos"

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #
def load_data(excel_path):
    """
    Load the participants Excel file, remove the excluded IDs and convert the age column to numeric.

    :param excel_path: Path to the Excel file with the participants report
    :return: DataFrame with the remaining participants and a numeric age column
    """
    df = pd.read_excel(excel_path, sheet_name=SHEET_NAME)
    df.columns = [str(c).strip() for c in df.columns]

    needed = [ID_COL, AGE_COL, SEX_COL]
    absent = [c for c in needed if c not in df.columns]
    if absent:
        sys.exit(
            f"Columns not found in Excel file: {absent}\n"
            f"Available columns: {list(df.columns)}"
        )

    mask = df[ID_COL].isin(IDS_TO_REMOVE)
    print(f"Participants in file: {len(df)}")
    print(f"Removed ({int(mask.sum())}): {sorted(df.loc[mask, ID_COL])}")
    not_found = [i for i in IDS_TO_REMOVE if i not in set(df[ID_COL])]
    if not_found:
        print(f"WARNING: IDs to remove that do not exist in the file: {not_found}")
    df = df[~mask].copy()
    print(f"Participants after removal: {len(df)}")

    df[AGE_COL] = _to_number(df[AGE_COL])
    return df


def draw_age_histogram(ages):
    """
    Draw the age histogram of all participants, with the count above each bar and a summary title (n, mean ± SD, range).

    :param ages: Series with the participants' ages in years (missing values are ignored)
    :return: Matplotlib figure with the age histogram
    """
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
    colors = [COLOR_MAIN for c in centers]
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

    fig.tight_layout(rect=(0, 0, 1, 1))
    return fig


def draw_sex_donut(n_female, n_male):
    """
    Draw a donut chart with the sex distribution, showing counts on the slices, the total in the center
    and percentages in the legend.

    :param n_female: Number of female participants
    :param n_male: Number of male participants
    :return: Matplotlib figure with the donut chart
    """
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


def main():
    """
    Run the full pipeline: load the participants data, print age and sex statistics, and save the
    age histogram and the sex donut chart. The Excel path can be overridden with the first command-line argument.

    :return: None
    """
    excel = sys.argv[1] if len(sys.argv) > 1 else EXCEL_PATH
    if len(sys.argv) > 1:
        globals()["EXCEL_PATH"] = excel
    _check_config()
    _setup_font()

    df = load_data(excel)

    ages = df[AGE_COL].dropna()
    print(f"\nAge: n = {len(ages)}, mean = {ages.mean():.1f} ± {ages.std(ddof=1):.1f}, "
          f"range = {ages.min():.0f}–{ages.max():.0f}")

    n_f = int((df[SEX_COL] == FEMALE_VALUE).sum())
    n_m = int((df[SEX_COL] == MALE_VALUE).sum())
    print(f"Sex: {LABEL_FEMALE} = {n_f}, {LABEL_MALE} = {n_m}")
    if n_f + n_m != len(df):
        print(f"WARNING: {len(df) - n_f - n_m} participant(s) with undefined sex "
              f"(neither {FEMALE_VALUE} nor {MALE_VALUE}).")
    print()

    _save(draw_age_histogram(df[AGE_COL]), "idades")
    _save(draw_sex_donut(n_f, n_m), "sexo")


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _check_config():
    """
    Check that no configuration constant was left with a placeholder value ("FILL_IN...") and exit if any was.

    :return: None
    """
    missing = [
        name
        for name, value in globals().items()
        if name.isupper() and isinstance(value, str) and value.startswith("FILL_IN")
    ]
    if missing:
        sys.exit("Please fill in the following at the top of the script:\n  - " + "\n  - ".join(sorted(missing)))


def _to_number(series):
    """
    Convert a Series to numeric, accepting decimal commas. Values that cannot be converted become NaN.

    :param series: Series to convert
    :return: Numeric Series
    """
    if series.dtype == object:
        series = series.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(series, errors="coerce")


def _setup_font():
    """
    Set the matplotlib font family to the first preferred font that is installed,
    falling back to DejaVu Sans.

    :return: None
    """
    preferred = ["Inter", "DM Sans", "Helvetica Neue", "Helvetica", "Arial"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = next((n for n in preferred if n in available), "DejaVu Sans")


def _save(fig, name):
    """
    Save a figure as PNG and SVG in OUTPUT_DIR (created if needed) and close it.

    :param fig: Matplotlib figure to save
    :param name: Name suffix of the output files (appended to OUTPUT_PREFIX)
    :return: None
    """
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        out = out_dir / f"{OUTPUT_PREFIX}_{name}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out.resolve()}")
    plt.close(fig)


if __name__ == "__main__":
    main()