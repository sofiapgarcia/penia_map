"""
Functions to generate radar charts for the Penia Map.

Available Functions
-------------------
[Public]
load_data(...): Load and clean participant data from an Excel file.
print_counts(...): Print participant counts by sex and age group.
load_profiles(...): Calculate group means as percentages of the young reference group.
draw_radar(...): Generate a diamond-shaped radar chart for one sex.
main(...): Run the complete pipeline and save the charts.

[Private]
_check_config(...): Validate the script configuration.
_selected_sexes(...): Determine which sexes to plot.
_norm_value(...): Normalize a single value for comparison.
_norm(...): Normalize the values in a Series.
_to_number(...): Convert a Series to numeric values.
_pick_font(...): Select an available font from the preferred list.
_diamond(...): Generate the vertices of a diamond.
_profile_vertices(...): Convert profile values into polygon vertices.
_ring_band(...): Create a diamond-shaped reference band.
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
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Patch, Polygon
from matplotlib.path import Path as MplPath

# =====================================================================
# 1) CONFIGURATION
# =====================================================================

EXCEL_PATH = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\report_participants_valids.xlsx"
SHEET_NAME = 0

# sex to plot
SEX_TO_PLOT = "both"   # "F" = women | "M" = men | "both" = generates both charts

# identification columns
ID_COL = "ID"
AGE_GROUP_COL = "Age group (MeSH)"
SEX_COL = "Gender"

# participants to exclude
IDS_TO_REMOVE = [4,14,17,22,24]

# values inside those columns
FEMALE_VALUE = 0      # value that represents women in SEX_COL
MALE_VALUE = 1        # value that represents men
GROUP_YOUNG = "Young Adult (18-24)"
GROUP_MIDDLE = "Middle Aged (45-64)"
GROUP_ELDERLY = "Aged (>=65)"

# --- metric columns (one per domain) ---
COL_MUSCLE_MASS = "Global muscle kg"        # sarcopenia   (kg)
COL_GRIP = "hi_values_max"                  # dynapenia    (kg)
COL_BMD = "Bone Mineral Density"            # osteopenia   (g/cm²)
COL_PEAK_POWER = "peak_power"               # powerpenia   (W)

# --- text shown in the chart (feel free to edit) ---
LABELS = {
    "F": dict(noun="mulheres", middle="Adultas (45-64)", elderly="Idosas (>=65)"),
    "M": dict(noun="homens", middle="Adultos (45-64)", elderly="Idosos (>=65)"),
}
LABEL_YOUNG = "jovens (18-24)"

# --- appearance ---
COLOR_MIDDLE = "#5B4DC4"   # purple
COLOR_ELDERLY = "#D1495B"  # red/coral, for contrast
SHOW_REF_BAND = False       # dashed grey band (reference range)
BAND_RANGE = (80, 100)     # band limits, in % of the young group
SHOW_VALUES_IN_LABELS = True  # show each group's % below the axis name
SHOW_RING_LABELS = True    # write 25%, 50%, 75%, 100% along the top axis
OUTPUT_PREFIX = "penia_map_radar"  # generates penia_map_radar_<mulheres|homens>.png and .svg

# =====================================================================
# 2) AXIS DEFINITIONS (order: top, right, bottom, left)
# =====================================================================

AXES = [
    dict(title="Sarcopenia", metric="Massa muscular", col=COL_MUSCLE_MASS),
    dict(title="Dinapenia", metric="Força de preensão manual", col=COL_GRIP),
    dict(title="Osteopenia", metric="Densidade mineral óssea", col=COL_BMD),
    dict(title="Powerpenia", metric="Pico de potência", col=COL_PEAK_POWER),
]

# =====================================================================
# 3) DATA PROCESSING
# =====================================================================

# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #
def _check_config():
    """
    Abort the script if any configuration constant was left as a placeholder.

    :return: None (exits with an error message if something is missing)
    """
    missing = [
        name
        for name, value in globals().items()
        if name.isupper() and isinstance(value, str) and value.startswith("FILL_IN")
    ]
    if missing:
        sys.exit(
            "The following settings must be filled in at the top of the script:\n  - "
            + "\n  - ".join(sorted(missing))
        )


def _selected_sexes():
    """
    Return the list of sexes to plot according to SEX_TO_PLOT.

    :return: List of sex codes: ["F"], ["M"] or ["F", "M"]
    """
    choice = str(SEX_TO_PLOT).strip().lower()
    if choice in ("f", "female", "women", "woman", "mulheres", "mulher"):
        return ["F"]
    if choice in ("m", "male", "men", "man", "homens", "homem"):
        return ["M"]
    if choice in ("both", "all", "ambos"):
        return ["F", "M"]
    sys.exit(f'Invalid SEX_TO_PLOT: "{SEX_TO_PLOT}". Use "F", "M" or "both".')


def _norm_value(v):
    """
    Normalize a value for comparison: 1, 1.0 and "1" become equal; case is ignored.

    :param v: Value to normalize (number or string)
    :return: Normalized lowercase string
    """
    if isinstance(v, (int, float, np.integer, np.floating)) and not pd.isna(v) and float(v).is_integer():
        v = int(v)
    return str(v).strip().lower()


def _norm(series):
    """
    Apply value normalization to every element of a Series.

    :param series: Input Series
    :return: Series of normalized strings
    """
    return series.map(_norm_value)


def _to_number(series):
    """
    Convert a Series to numeric, accepting a decimal comma (e.g. "41,4").

    :param series: Input Series
    :return: Numeric Series (unparseable values become NaN)
    """
    if series.dtype == object:
        series = series.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(series, errors="coerce")


def load_data(excel_path):
    """
    Load the Excel file, drop excluded IDs and convert metric columns to numbers.

    :param excel_path: Path to the Excel file
    :return: Cleaned DataFrame with all participants
    """
    df = pd.read_excel(excel_path, sheet_name=SHEET_NAME)
    df.columns = [str(c).strip() for c in df.columns]

    if ID_COL in df.columns:
        df = df[~df[ID_COL].isin(IDS_TO_REMOVE)]
    elif IDS_TO_REMOVE:
        sys.exit(f"ID column '{ID_COL}' not found, so IDS_TO_REMOVE cannot be applied.")

    needed = [AGE_GROUP_COL, SEX_COL] + [a["col"] for a in AXES]
    absent = [c for c in needed if c not in df.columns]
    if absent:
        sys.exit(
            f"Columns not found in the Excel file: {absent}\n"
            f"Available columns: {list(df.columns)}"
        )
    for a in AXES:
        df[a["col"]] = _to_number(df[a["col"]])
    return df


def print_counts(df_all):
    """
    Print n per age group and in total, for women and for men.

    :param df_all: DataFrame with all participants
    :return: None
    """
    metric_cols = [a["col"] for a in AXES]
    complete = df_all[metric_cols].notna().all(axis=1)
    sex_norm = _norm(df_all[SEX_COL])
    group_norm = _norm(df_all[AGE_GROUP_COL])
    group_order = [GROUP_YOUNG, GROUP_MIDDLE, GROUP_ELDERLY]

    for sex in ("F", "M"):
        noun = LABELS[sex]["noun"].upper()
        value = FEMALE_VALUE if sex == "F" else MALE_VALUE
        in_sex = sex_norm == _norm_value(value)
        print(f"\n===== {noun}: PARTICIPANTS =====")
        print(f"{'Age group':<28}{'n':>5}{'with 4 metrics':>18}")
        for g in group_order:
            m = in_sex & (group_norm == _norm_value(g))
            print(f"{g:<28}{int(m.sum()):>5}{int((m & complete).sum()):>18}")
        others = in_sex & ~group_norm.isin([_norm_value(g) for g in group_order])
        if others.any():
            print(f"{'Other groups / empty':<28}{int(others.sum()):>5}{int((others & complete).sum()):>18}")
        print("-" * 51)
        print(f"{'TOTAL':<28}{int(in_sex.sum()):>5}{int((in_sex & complete).sum()):>18}")

    known = sex_norm.isin([_norm_value(FEMALE_VALUE), _norm_value(MALE_VALUE)])
    print(f"\nTotal participants in file: {len(df_all)}")
    if (~known).any():
        print(f"Sex not defined (neither {FEMALE_VALUE} nor {MALE_VALUE}): {int((~known).sum())}")


def load_profiles(df_all, sex_value, noun):
    """
    Compute the profiles (% of the young group's mean) for one sex.

    :param df_all: DataFrame with all participants
    :param sex_value: Value in SEX_COL that identifies the sex
    :param noun: Plural noun for the sex, used in messages (e.g. "women")
    :return: Tuple (profiles dict, array of reference means, n of the young group)
    """
    df = df_all[_norm(df_all[SEX_COL]) == _norm_value(sex_value)].copy()
    if df.empty:
        sys.exit(
            f"No rows with {SEX_COL} = {sex_value!r} ({noun}).\n"
            f"Existing values in '{SEX_COL}': {sorted(df_all[SEX_COL].astype(str).unique())}"
        )

    groups = {}
    for key, label in [
        ("young", GROUP_YOUNG),
        ("middle", GROUP_MIDDLE),
        ("elderly", GROUP_ELDERLY),
    ]:
        sub = df[_norm(df[AGE_GROUP_COL]) == _norm_value(label)]
        if sub.empty:
            sys.exit(
                f"Group '{label}' not found in '{AGE_GROUP_COL}' ({noun}).\n"
                f"Existing values: {sorted(df[AGE_GROUP_COL].astype(str).unique())}"
            )
        groups[key] = sub

    ref_means = np.array([groups["young"][a["col"]].mean() for a in AXES])
    profiles = {}
    for key in ("middle", "elderly"):
        means = np.array([groups[key][a["col"]].mean() for a in AXES])
        profiles[key] = dict(
            pct=means / ref_means * 100,
            raw=means,
            n=len(groups[key]),
        )
    return profiles, ref_means, len(groups["young"])


# =====================================================================
# 4) DRAWING
# =====================================================================


def _pick_font():
    """
    Pick the first available font from a preferred list.

    :return: Font family name (falls back to "DejaVu Sans")
    """
    preferred = ["Inter", "DM Sans", "Helvetica Neue", "Helvetica", "Arial"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            return name
    return "DejaVu Sans"


def _diamond(r):
    """
    Build the vertices of a diamond (top, right, bottom, left).

    :param r: Radius of the diamond
    :return: (4, 2) array of vertices
    """
    return np.array([[0, r], [r, 0], [0, -r], [-r, 0]], dtype=float)


def _profile_vertices(values):
    """
    Convert four axis values into polygon vertices (top, right, bottom, left).

    :param values: Sequence of four values, one per axis
    :return: (4, 2) array of vertices
    """
    v = values
    return np.array([[0, v[0]], [v[1], 0], [0, -v[2]], [-v[3], 0]], dtype=float)


def _ring_band(r_in, r_out, **kwargs):
    """
    Build a diamond-shaped ring (outer diamond with an inner hole).

    :param r_in: Inner radius
    :param r_out: Outer radius
    :param kwargs: Extra keyword arguments passed to PathPatch
    :return: PathPatch representing the ring
    """
    outer = _diamond(r_out)
    inner = _diamond(r_in)[::-1]  # reversed direction -> "hole"
    verts = np.vstack([outer, outer[:1], inner, inner[:1]])
    codes = (
        [MplPath.MOVETO] + [MplPath.LINETO] * 3 + [MplPath.CLOSEPOLY]
        + [MplPath.MOVETO] + [MplPath.LINETO] * 3 + [MplPath.CLOSEPOLY]
    )
    return PathPatch(MplPath(verts, codes), **kwargs)

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def draw_radar(profiles, n_young, lab):
    """
    Draw the diamond-shaped radar chart for one sex.

    :param profiles: Dict with "middle" and "elderly" profiles (pct, raw, n)
    :param n_young: Number of participants in the young reference group
    :param lab: Dict with the texts for the chosen sex (noun, middle, elderly)
    :return: Matplotlib Figure
    """
    font = _pick_font()
    plt.rcParams["font.family"] = font

    max_val = max(p["pct"].max() for p in profiles.values())
    rmax = 25 * int(np.ceil(max(100, max_val) / 25))
    rings = list(range(25, rmax + 1, 25))

    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-rmax * 1.95, rmax * 1.95)
    ax.set_ylim(-rmax * 1.5, rmax * 1.5)

    grid_color = "#D9D9D9"
    text_gray = "#6B6B6B"

    # grey reference band
    if SHOW_REF_BAND:
        ax.add_patch(
            _ring_band(
                BAND_RANGE[0], BAND_RANGE[1],
                facecolor="#E8E8E8", edgecolor="#9A9A9A",
                linewidth=1.0, linestyle=(0, (4, 3)), alpha=0.9, zorder=1,
            )
        )

    # diamond rings
    for r in rings:
        ax.add_patch(
            Polygon(_diamond(r), closed=True, fill=False, edgecolor=grid_color,
                    linewidth=1.1 if r == 100 else 0.8, zorder=2)
        )
        if SHOW_RING_LABELS:
            ax.text(rmax * 0.02, r, f"{r}%", ha="left", va="center", fontsize=8.5,
                    color=text_gray, zorder=3,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.85))

    # axes
    for vx, vy in _diamond(rmax):
        ax.plot([0, vx], [0, vy], color=grid_color, linewidth=0.8, zorder=2)

    # polygons
    styles = {
        "middle": dict(color=COLOR_MIDDLE),
        "elderly": dict(color=COLOR_ELDERLY),
    }
    for key in ("middle", "elderly"):
        verts = _profile_vertices(profiles[key]["pct"])
        c = styles[key]["color"]
        ax.add_patch(Polygon(verts, closed=True, facecolor=c, alpha=0.16,
                             edgecolor="none", zorder=4))
        ax.add_patch(Polygon(verts, closed=True, fill=False, edgecolor=c,
                             linewidth=2.2, joinstyle="round", zorder=5))
        ax.scatter(verts[:, 0], verts[:, 1], s=55, color=c,
                   edgecolor="white", linewidth=1.2, zorder=6)

    # axis labels
    tips = _diamond(rmax)
    pad = 12  # points
    for i, (a, (tx, ty)) in enumerate(zip(AXES, tips)):
        lines = [(a["title"], dict(fontsize=13, fontweight="bold", color="#111111")),
                 (a["metric"], dict(fontsize=10, color=text_gray))]
        if SHOW_VALUES_IN_LABELS:
            m = profiles["middle"]["pct"][i]
            e = profiles["elderly"]["pct"][i]
            lines.append((f"{lab['middle'].lower()} {m:.0f}% · {lab['elderly'].lower()} {e:.0f}%",
                          dict(fontsize=10, color=text_gray)))

        # position of each line (offsets in points) depending on the side
        if i == 0:      # top: stack upwards
            offs = [(0, 6 + 14 * (len(lines) - 1 - k)) for k in range(len(lines))]
            kw = dict(ha="center", va="bottom")
        elif i == 2:    # bottom: stack downwards
            offs = [(0, -pad + -15 * k) for k in range(len(lines))]
            kw = dict(ha="center", va="top")
        else:           # right / left: block vertically centered
            sign = 1 if i == 1 else -1
            total = len(lines)
            offs = [(sign * pad, 15 * ((total - 1) / 2 - k)) for k in range(total)]
            kw = dict(ha="left" if i == 1 else "right", va="center")

        for (txt, style), off in zip(lines, offs):
            ax.annotate(txt, xy=(tx, ty), xytext=off, textcoords="offset points",
                        zorder=7, **kw, **style)

    # legend
    handles = [
        Line2D([0], [0], color=COLOR_MIDDLE, marker="o", markersize=7, linewidth=2.2,
               markeredgecolor="white", label=f"{lab['middle']} (n = {profiles['middle']['n']})"),
        Line2D([0], [0], color=COLOR_ELDERLY, marker="o", markersize=7, linewidth=2.2,
               markeredgecolor="white", label=f"{lab['elderly']} (n = {profiles['elderly']['n']})"),
    ]
    if SHOW_REF_BAND:
        handles.append(Patch(facecolor="#E8E8E8", edgecolor="#9A9A9A", linestyle="--",
                             label=f"Faixa {BAND_RANGE[0]}–{BAND_RANGE[1]}%"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=10.5, bbox_to_anchor=(0.5, 0.06))
    fig.text(0.5, 0.035,
             f"100% = média de {lab['noun']} {LABEL_YOUNG} (n = {n_young})",
             ha="center", fontsize=9, color=text_gray)
    return fig


# =====================================================================
# 5) EXECUTION
# =====================================================================


def main():
    """
    Run the full pipeline: load data, print summaries, draw and save the charts.

    :return: None
    """
    excel = sys.argv[1] if len(sys.argv) > 1 else EXCEL_PATH
    if len(sys.argv) > 1:
        globals()["EXCEL_PATH"] = excel
    _check_config()

    df_all = load_data(excel)
    print_counts(df_all)
    sex_values = {"F": FEMALE_VALUE, "M": MALE_VALUE}

    for sex in _selected_sexes():
        lab = LABELS[sex]
        profiles, ref_means, n_young = load_profiles(df_all, sex_values[sex], lab["noun"])

        print(f"\n===== {lab['noun'].upper()} =====")
        print(f"Reference ({lab['noun']} {LABEL_YOUNG}, n = {n_young}):")
        for a, r in zip(AXES, ref_means):
            print(f"  {a['title']:<11} {a['metric']:<26} mean = {r:.3f}")
        for key in ("middle", "elderly"):
            p = profiles[key]
            print(f"\n{lab[key]} (n = {p['n']}):")
            for a, raw, pct in zip(AXES, p["raw"], p["pct"]):
                print(f"  {a['title']:<11} mean = {raw:.3f}  ->  {pct:.1f}%")

        fig = draw_radar(profiles, n_young, lab)
        for ext in ("png", "svg"):
            out = Path(f"{OUTPUT_PREFIX}_{lab['noun']}.{ext}")
            fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"\nSaved: {out.resolve()}")
        plt.close(fig)


if __name__ == "__main__":
    main()