IDS_TO_REMOVE = [4,14,17,22,24]

"""
Penia Map - gráfico radar (4 eixos, formato diamante)

Cada eixo mostra a média de um grupo em % da média do grupo JOVEM do mesmo
sexo (100% = jovem). São desenhados dois polígonos: meia-idade e idosos.
O sexo a representar escolhe-se em SEX_TO_PLOT ("F", "M" ou "both").

Uso:
    python graph_penia_map.py                 # usa EXCEL_PATH definido abaixo
    python graph_penia_map.py dados.xlsx      # ou passa o ficheiro na linha de comandos

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
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Patch, Polygon
from matplotlib.path import Path as MplPath

# =====================================================================
# 1) CONFIGURAÇÃO  ->  PREENCHE AQUI (tudo o que começa por "PREENCHER")
# =====================================================================

EXCEL_PATH = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\report_participants_valids.xlsx"
SHEET_NAME = 0  # 0 = primeira folha, ou o nome da folha, ex.: "Dados"

# --- sexo a representar ---
SEX_TO_PLOT = "both"   # "F" = mulheres | "M" = homens | "both" = gera os dois gráficos

# --- colunas de identificação (nome EXATO do cabeçalho no Excel) ---
AGE_GROUP_COL = "Age group (MeSH)"
SEX_COL = "Gender"

# --- valores dentro dessas colunas (não distingue maiúsculas/minúsculas) ---
FEMALE_VALUE = 0      # valor que representa as mulheres na coluna SEX_COL
MALE_VALUE = 1        # valor que representa os homens
GROUP_YOUNG = "Young Adult (18-24)"
GROUP_MIDDLE = "Middle Aged (45-64)"
GROUP_ELDERLY = "Aged (>=65)"

# --- colunas das métricas (uma por domínio) ---
COL_MUSCLE_MASS = "Global muscle kg"        # sarcopenia   (kg)
COL_GRIP = "hi_values_max"                  # dinapenia    (kg)
COL_BMD = "Bone Mineral Density"            # osteopenia   (g/cm²)
COL_PEAK_POWER = "peak_power"               # powerpenia   (W)

# --- texto que aparece no gráfico (podes traduzir/alterar) ---
LABELS = {
    "F": dict(noun="mulheres", middle="Adultas (45-64)", elderly="Idosas (>=65)"),
    "M": dict(noun="homens", middle="Adultos (45-64)", elderly="Idosos (>=65)"),
}
LABEL_YOUNG = "jovens (18-24)"

# --- aparência ---
COLOR_MIDDLE = "#5B4DC4"   # roxo (como no exemplo)
COLOR_ELDERLY = "#D1495B"  # vermelho/coral, para contrastar
SHOW_REF_BAND = False       # banda cinzenta tracejada (faixa de referência)
BAND_RANGE = (80, 100)     # limites da banda, em % do jovem
SHOW_VALUES_IN_LABELS = True  # mostra os % de cada grupo por baixo do nome do eixo
SHOW_RING_LABELS = True    # escreve 25%, 50%, 75%, 100% ao longo do eixo de cima
OUTPUT_PREFIX = "penia_map_radar"  # gera penia_map_radar_<mulheres|homens>.png e .svg

# =====================================================================
# 2) DEFINIÇÃO DOS EIXOS (ordem: topo, direita, baixo, esquerda)
# =====================================================================

AXES = [
    dict(title="Sarcopenia", metric="Massa muscular", col=COL_MUSCLE_MASS),
    dict(title="Dinapenia", metric="Força de preensão manual", col=COL_GRIP),
    dict(title="Osteopenia", metric="Densidade mineral óssea", col=COL_BMD),
    dict(title="Powerpenia", metric="Pico de potência", col=COL_PEAK_POWER),
]

# =====================================================================
# 3) PROCESSAMENTO DOS DADOS
# =====================================================================


def _check_config():
    missing = [
        name
        for name, value in globals().items()
        if name.isupper() and isinstance(value, str) and value.startswith("PREENCHER")
    ]
    if missing:
        sys.exit(
            "Falta preencher no topo do script:\n  - " + "\n  - ".join(sorted(missing))
        )


def _selected_sexes():
    """Devolve a lista de sexos a desenhar: ["F"], ["M"] ou ["F", "M"]."""
    choice = str(SEX_TO_PLOT).strip().lower()
    if choice in ("f", "female", "mulheres", "mulher"):
        return ["F"]
    if choice in ("m", "male", "homens", "homem"):
        return ["M"]
    if choice in ("both", "ambos", "all"):
        return ["F", "M"]
    sys.exit(f'SEX_TO_PLOT inválido: "{SEX_TO_PLOT}". Usa "F", "M" ou "both".')


def _norm_value(v):
    """Normaliza um valor para comparação: 1, 1.0 e "1" ficam iguais; ignora maiúsculas."""
    if isinstance(v, (int, float, np.integer, np.floating)) and not pd.isna(v) and float(v).is_integer():
        v = int(v)
    return str(v).strip().lower()


def _norm(series):
    return series.map(_norm_value)


def _to_number(series):
    """Converte para número, aceitando vírgula decimal ("41,4")."""
    if series.dtype == object:
        series = series.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(series, errors="coerce")


def load_data(excel_path):
    df = pd.read_excel(excel_path, sheet_name=SHEET_NAME)
    df = df[~df["ID"].isin(IDS_TO_REMOVE)]
    df.columns = [str(c).strip() for c in df.columns]

    needed = [AGE_GROUP_COL, SEX_COL] + [a["col"] for a in AXES]
    absent = [c for c in needed if c not in df.columns]
    if absent:
        sys.exit(
            f"Colunas não encontradas no Excel: {absent}\n"
            f"Colunas disponíveis: {list(df.columns)}"
        )
    for a in AXES:
        df[a["col"]] = _to_number(df[a["col"]])
    return df


def print_counts(df_all):
    """Imprime n por faixa etária e total, para mulheres e para homens."""
    metric_cols = [a["col"] for a in AXES]
    complete = df_all[metric_cols].notna().all(axis=1)
    sex_norm = _norm(df_all[SEX_COL])
    group_norm = _norm(df_all[AGE_GROUP_COL])
    group_order = [GROUP_YOUNG, GROUP_MIDDLE, GROUP_ELDERLY]

    for sex in ("F", "M"):
        noun = LABELS[sex]["noun"].upper()
        value = FEMALE_VALUE if sex == "F" else MALE_VALUE
        in_sex = sex_norm == _norm_value(value)
        print(f"\n===== {noun}: PARTICIPANTES =====")
        print(f"{'Faixa etária':<28}{'n':>5}{'com 4 métricas':>18}")
        for g in group_order:
            m = in_sex & (group_norm == _norm_value(g))
            print(f"{g:<28}{int(m.sum()):>5}{int((m & complete).sum()):>18}")
        others = in_sex & ~group_norm.isin([_norm_value(g) for g in group_order])
        if others.any():
            print(f"{'Outras faixas / vazio':<28}{int(others.sum()):>5}{int((others & complete).sum()):>18}")
        print("-" * 51)
        print(f"{'TOTAL':<28}{int(in_sex.sum()):>5}{int((in_sex & complete).sum()):>18}")

    known = sex_norm.isin([_norm_value(FEMALE_VALUE), _norm_value(MALE_VALUE)])
    print(f"\nTotal de participantes no ficheiro: {len(df_all)}")
    if (~known).any():
        print(f"Sem sexo definido (nem {FEMALE_VALUE} nem {MALE_VALUE}): {int((~known).sum())}")


def load_profiles(df_all, sex_value, noun):
    """Calcula os perfis (% do grupo jovem) para um sexo."""
    df = df_all[_norm(df_all[SEX_COL]) == _norm_value(sex_value)].copy()
    if df.empty:
        sys.exit(
            f"Nenhuma linha com {SEX_COL} = {sex_value!r} ({noun}).\n"
            f"Valores existentes em '{SEX_COL}': {sorted(df_all[SEX_COL].astype(str).unique())}"
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
                f"Grupo '{label}' não encontrado em '{AGE_GROUP_COL}' ({noun}).\n"
                f"Valores existentes: {sorted(df[AGE_GROUP_COL].astype(str).unique())}"
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
# 4) DESENHO
# =====================================================================


def _pick_font():
    preferred = ["Inter", "DM Sans", "Helvetica Neue", "Helvetica", "Arial"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            return name
    return "DejaVu Sans"


def _diamond(r):
    """Vértices: topo, direita, baixo, esquerda."""
    return np.array([[0, r], [r, 0], [0, -r], [-r, 0]], dtype=float)


def _profile_vertices(values):
    v = values
    return np.array([[0, v[0]], [v[1], 0], [0, -v[2]], [-v[3], 0]], dtype=float)


def _ring_band(r_in, r_out, **kwargs):
    outer = _diamond(r_out)
    inner = _diamond(r_in)[::-1]  # sentido contrário -> "buraco"
    verts = np.vstack([outer, outer[:1], inner, inner[:1]])
    codes = (
        [MplPath.MOVETO] + [MplPath.LINETO] * 3 + [MplPath.CLOSEPOLY]
        + [MplPath.MOVETO] + [MplPath.LINETO] * 3 + [MplPath.CLOSEPOLY]
    )
    return PathPatch(MplPath(verts, codes), **kwargs)


def draw_radar(profiles, n_young, lab):
    """lab = dict(noun=..., middle=..., elderly=...) com os textos do sexo escolhido."""
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

    # banda cinzenta de referência
    if SHOW_REF_BAND:
        ax.add_patch(
            _ring_band(
                BAND_RANGE[0], BAND_RANGE[1],
                facecolor="#E8E8E8", edgecolor="#9A9A9A",
                linewidth=1.0, linestyle=(0, (4, 3)), alpha=0.9, zorder=1,
            )
        )

    # anéis em diamante
    for r in rings:
        ax.add_patch(
            Polygon(_diamond(r), closed=True, fill=False, edgecolor=grid_color,
                    linewidth=1.1 if r == 100 else 0.8, zorder=2)
        )
        if SHOW_RING_LABELS:
            ax.text(rmax * 0.02, r, f"{r}%", ha="left", va="center", fontsize=8.5,
                    color=text_gray, zorder=3,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.85))

    # eixos
    for vx, vy in _diamond(rmax):
        ax.plot([0, vx], [0, vy], color=grid_color, linewidth=0.8, zorder=2)

    # polígonos
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

    # rótulos dos eixos
    tips = _diamond(rmax)
    pad = 12  # pontos
    for i, (a, (tx, ty)) in enumerate(zip(AXES, tips)):
        lines = [(a["title"], dict(fontsize=13, fontweight="bold", color="#111111")),
                 (a["metric"], dict(fontsize=10, color=text_gray))]
        if SHOW_VALUES_IN_LABELS:
            m = profiles["middle"]["pct"][i]
            e = profiles["elderly"]["pct"][i]
            lines.append((f"{lab['middle'].lower()} {m:.0f}% · {lab['elderly'].lower()} {e:.0f}%",
                          dict(fontsize=10, color=text_gray)))

        # posição de cada linha (offsets em pontos) consoante o lado
        if i == 0:      # topo: empilha para cima
            offs = [(0, 6 + 14 * (len(lines) - 1 - k)) for k in range(len(lines))]
            kw = dict(ha="center", va="bottom")
        elif i == 2:    # baixo: empilha para baixo
            offs = [(0, -pad + -15 * k) for k in range(len(lines))]
            kw = dict(ha="center", va="top")
        else:           # direita / esquerda: bloco centrado na vertical
            sign = 1 if i == 1 else -1
            total = len(lines)
            offs = [(sign * pad, 15 * ((total - 1) / 2 - k)) for k in range(total)]
            kw = dict(ha="left" if i == 1 else "right", va="center")

        for (txt, style), off in zip(lines, offs):
            ax.annotate(txt, xy=(tx, ty), xytext=off, textcoords="offset points",
                        zorder=7, **kw, **style)

    # legenda
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
# 5) EXECUÇÃO
# =====================================================================


def main():
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

        # resumo na consola (útil para confirmar os números)
        print(f"\n===== {lab['noun'].upper()} =====")
        print(f"Referência ({lab['noun']} {LABEL_YOUNG}, n = {n_young}):")
        for a, r in zip(AXES, ref_means):
            print(f"  {a['title']:<11} {a['metric']:<26} média = {r:.3f}")
        for key in ("middle", "elderly"):
            p = profiles[key]
            print(f"\n{lab[key]} (n = {p['n']}):")
            for a, raw, pct in zip(AXES, p["raw"], p["pct"]):
                print(f"  {a['title']:<11} média = {raw:.3f}  ->  {pct:.1f}%")

        fig = draw_radar(profiles, n_young, lab)
        for ext in ("png", "svg"):
            out = Path(f"{OUTPUT_PREFIX}_{lab['noun']}.{ext}")
            fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"\nGuardado: {out.resolve()}")
        plt.close(fig)


if __name__ == "__main__":
    main()