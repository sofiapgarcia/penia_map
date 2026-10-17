"""
Generate one PDF report per participant from an Excel workbook with two sheets:
  - "Classified data"      -> one row per participant
  - "Means group x sex"    -> reference means per age group (MeSH) and sex

Each of the 4 main measures is classified as Baixo / Médio / Alto using a z-score
(participant vs. the mean of their own sex + age group):
    z < -Z_CUT        -> Baixo
    -Z_CUT <= z <= +Z_CUT -> Médio
    z > +Z_CUT        -> Alto
The mean comes from the reference sheet; the standard deviation is computed from the
participants sheet (same age group + sex; falls back to the whole sex if the group is small).

Dependencies: pip install pandas jinja2 weasyprint openpyxl
"""
import math
import os
os.environ["WEASYPRINT_DLL_DIRECTORIES"] = r"C:\msys64\ucrt64\bin"
import base64
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML

# =====================================================================
# 1) CONFIGURATION
# =====================================================================
BASE = Path(r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\results")
LOGO_FACULTY = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\results\participant_reports\imagens\nova_fct.png"
LOGO_LAB = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\results\participant_reports\imagens\libphys.png"
LOGO_PROJECT = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\results\participant_reports\imagens\logos.png"
EXCEL_FILE = BASE / "means_report_participant.xlsx"
PARTICIPANTS_SHEET = "Classified data"
REFERENCE_SHEET = "Means group x sex"
OUTPUT_DIR = BASE / "participant_reports"
ICONS_DIR = BASE / "participant_reports" / "imagens"   # sarcopenia.png, dynapenia.png, osteopenia.png, powerpenia.png
TEMPLATE_FILE = BASE / "participant_reports" / "template_report_2.html"

# None -> generate for ALL participants.
# Otherwise: exactly the value in the "Name" column (e.g. "Sujeito 125" or "125").
TEST_PARTICIPANT = "Maria Manuela Neto Poeiras Garcia"

# Column names in the participants sheet (left = internal key, right = YOUR column name)
COL = {
    "name": "Name",
    "date": "Date",
    "sex": "Sex",                                   # <-- NEW: check this column name
    "chrono_age": "Age (y)",
    "bio_age": "Estimated metabolic age",
    "age_group": "Age group (MeSH)",
    # main measures
    "muscle_mass": "Global muscle kg",
    "grip_strength": "hi_values_max",
    "bmd": "Bone Mineral Density",
    "peak_power": "peak_power",
    # Tanita scale - anthropometrics
    "height": "Height(cm)", "weight": "Body mass(kg)", "bmi": "Body mass index (BMI)",
    # muscle mass by segment
    "mm_total": "Global muscle kg",
    "mm_right_arm": "Arm muscle (right) kg",
    "mm_left_arm": "Arm muscle (left) kg",
    "mm_right_leg": "Leg muscle (right) kg",
    "mm_left_leg": "Leg muscle (left) kg",
    "mm_trunk": "Torso muscle kg",
    # body fat by segment
    "fat_total": "Global fat %",
    "fat_right_arm": "Arm fat (right) %",
    "fat_left_arm": "Arm fat (left) %",
    "fat_right_leg": "Leg fat (right) %",
    "fat_left_leg": "Leg fat (left) %",
    "fat_trunk": "Torso fat %",
    # others
    "bone_mass": "Estimated bone mass kg",
    "water": "Global body water %",
    "calories": "Daily calorie intake (DCI)",
}

# Column names in the reference sheet (long format: one row per age group x sex)
REF_COL = {
    "group": "Age group (MeSH)",
    "sex": "Sex",
    "muscle_mass": "Global muscle kg",
    "grip_strength": "hi_values_max",
    "bmd": "Bone Mineral Density",
    "peak_power": "peak_power",
}
SEX_FEMALE, SEX_MALE = "Female", "Male"

# Accepted values in the "Sex" column of the participants sheet (lowercase) -> SEX_FEMALE / SEX_MALE
SEX_MAP = {
    "female": SEX_FEMALE, "f": SEX_FEMALE, "feminino": SEX_FEMALE, "mulher": SEX_FEMALE,
    "male": SEX_MALE, "m": SEX_MALE, "masculino": SEX_MALE, "homem": SEX_MALE,
}
SEX_DISPLAY = {SEX_FEMALE: "Mulheres", SEX_MALE: "Homens"}
SEX_ADJ = {
    SEX_FEMALE: "feminino",
    SEX_MALE: "masculino"
}
info_know_more_sex = {
    SEX_FEMALE: "informada",
    SEX_MALE: "informado"
}
welcome_sex = {
    SEX_FEMALE: "Bem-vinda",
    SEX_MALE: "Bem-vindo"
}
# Text shown in the report's table header for each MeSH age group
GROUP_DISPLAY = {
    "Young Adult (18-24)": "Jovem Adulto: 18-24 anos",
    "Adult (25-44)": "Adulto: 25-44 anos",
    "Middle Aged (45-64)": "Adulto: 45–64 anos",
    "Aged (>=65)": "Idoso: ≥65 anos",
}

# ---- Classification settings ----
MIN_N_FOR_TERCILES = 5   # min. participants in a (group, sex) cell; otherwise terciles use the whole sex
DECIMAL_COMMA = False  # True -> 41,4 kg ; False -> 41.4 kg

# The 4 measures, in radar order: top, right, bottom, left
MEASURES = [
    dict(key="muscle_mass", penia="Sarcopenia", icon="sarcopenia", name="Massa muscular",
         short="Massa muscular", det="a massa muscular", unit="kg", dec=1, trim=True,
         meaning="Quantidade de músculo no corpo"),
    dict(key="grip_strength", penia="Dinapenia", icon="dynapenia", name="Força de preensão isométrica",
         short="Força de preensão isométrica", det="a força de preensão", unit="kg", dec=1, trim=True,
         meaning="A força máxima que conseguiu exercer ao apertar o dinamómetro"),
    dict(key="bmd", penia="Osteopenia", icon="osteopenia", name="Densidade mineral óssea (estimada)",
         short="Densidade mineral óssea", det="a densidade mineral óssea", unit="g/cm²", dec=3, trim=False,
         meaning="Reflete a robustez dos ossos"),
    dict(key="peak_power", penia="Powerpenia", icon="powerpenia", name="Pico de potência",
         short="Pico de potência", det="o pico de potência", unit="W", dec=2, trim=False,
         meaning="A rapidez com que consegue gerar força com a mão."),
]
LEVELS = {"low": "Baixo", "mid": "Médio", "high": "Alto"}
# =====================================================================


# ---------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------
def fmt(x, decimals=1):
    """Format a number without useless zeros (41.40 -> 41.4; 7.0 -> 7)."""
    if pd.isna(x):
        return "—"
    s = f"{float(x):.{decimals}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s.replace(".", ",") if DECIMAL_COMMA else s


def fmt_fixed(x, decimals):
    """Format with a fixed number of decimals (e.g. 1.147, 14.50)."""
    if pd.isna(x):
        return "—"
    s = f"{float(x):.{decimals}f}"
    return s.replace(".", ",") if DECIMAL_COMMA else s


def fmt_measure(m, x):
    """Format a participant value for one of the 4 main measures (no unit)."""
    return fmt(x, m["dec"]) if m["trim"] else fmt_fixed(x, m["dec"])


def join_pt(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " e " + items[-1]


def icon_uri(stem):
    path = ICONS_DIR / f"{stem}.png"
    return path.resolve().as_uri() if path.exists() else ""
def image_uri(path):
    """Return a file URI for an image path."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    return path.resolve().as_uri()


# ---------------------------------------------------------------------
# Reference data + classification
# ---------------------------------------------------------------------
def load_reference(df):
    """Fill merged/blank age-group cells downward and clean labels."""
    df = df.copy()
    df[REF_COL["group"]] = df[REF_COL["group"]].ffill().astype(str).str.strip()
    df[REF_COL["sex"]] = df[REF_COL["sex"]].astype(str).str.strip()
    return df


def reference_row(refs, group, sex):
    match = refs[(refs[REF_COL["group"]] == str(group).strip()) & (refs[REF_COL["sex"]] == sex)]
    if match.empty:
        raise ValueError(f"No reference row for age group '{group}' / sex '{sex}'.")
    return match.iloc[0]


def normalize_sex(value):
    key = str(value).strip().lower()
    if key in SEX_MAP:
        return SEX_MAP[key]
    raise ValueError(f"Unrecognised value '{value}' in column '{COL['sex']}' - add it to SEX_MAP.")

def years_pt(n):
    """'ano' for ±1, 'anos' for everything else (including 0)."""
    return "ano" if abs(int(n)) == 1 else "anos"

def build_samples(participants):
    """Values of every measure per (age group, sex) and per sex. Use ALL participants (before filtering)."""
    df = participants.copy()
    df["_group"] = df[COL["age_group"]].astype(str).str.strip()
    df["_sex"] = df[COL["sex"]].map(lambda v: SEX_MAP.get(str(v).strip().lower()))
    by_group, by_sex = {}, {}
    for m in MEASURES:
        df["_v"] = pd.to_numeric(df[COL[m["key"]]], errors="coerce")
        for (grp, sex), s in df.groupby(["_group", "_sex"])["_v"]:
            s = s.dropna()
            if len(s) >= MIN_N_FOR_TERCILES:
                by_group[(grp, sex, m["key"])] = s.to_numpy()
        for sex, s in df.groupby("_sex")["_v"]:
            by_sex[(sex, m["key"])] = s.dropna().to_numpy()
    return by_group, by_sex


def get_sample(samples, group, sex, key):
    by_group, by_sex = samples
    arr = by_group.get((group, sex, key))
    if arr is None:
        arr = by_sex.get((sex, key))
    if arr is None or len(arr) < 3:
        raise ValueError(f"Not enough data to compute terciles for '{key}' ({sex}).")
    return arr


def classify(value, sample, key):
    """Percentile of the value within its group -> tercile (Baixo / Médio / Alto)."""
    if pd.isna(value):
        raise ValueError(f"Missing value for '{key}'.")
    v = float(value)
    pct = ((sample < v).sum() + 0.5 * (sample == v).sum()) / len(sample)    # 0..1
    level = "low" if pct < 1 / 3 else "high" if pct > 2 / 3 else "mid"
    return dict(pct=pct, level=level, label=LEVELS[level], frac=pct, pos=round(pct * 100, 1))

# ---------------------------------------------------------------------
# Visual pieces
# ---------------------------------------------------------------------
def build_timeline(chrono, bio):
    """Scale centred on the chronological age (always at 50%), with one tick per year."""
    half = max(10, math.ceil((abs(bio - chrono) + 3) / 5) * 5)     # years shown on each side of the centre
    pos = lambda age: 50 + (age - chrono) / half * 44              # 6% ... 94%
    ticks = []
    for age in range(chrono - half, chrono + half + 1):
        if age < 0:
            continue
        major = age % 5 == 0
        near_marker = abs(age - chrono) < 1.5 or abs(age - bio) < 1.5
        ticks.append(dict(pos=round(pos(age), 2), major=major,
                          label=str(age) if major and not near_marker else ""))
    pb = pos(bio)
    return dict(chrono=50, bio=round(pb, 1), left=round(min(50, pb), 1), width=round(abs(pb - 50), 1), ticks=ticks)

def build_space_uri():
    """Earth + trajectory + satellite illustration as an SVG data-URI."""
    parts = (
        # estrelas
        '<circle cx="62" cy="52" r="1.2" fill="#8c9bd0"/>'
        '<circle cx="82" cy="40" r="1" fill="#8c9bd0"/>'
        '<circle cx="48" cy="12" r="1" fill="#8c9bd0"/>'
        '<circle cx="112" cy="50" r="1.3" fill="#8c9bd0"/>'
        '<path d="M70 12 l1.2 3 3 1.2 -3 1.2 -1.2 3 -1.2 -3 -3 -1.2 3 -1.2 z" fill="#ccf79a"/>'
        '<path d="M108 30 l0.9 2.2 2.2 0.9 -2.2 0.9 -0.9 2.2 -0.9 -2.2 -2.2 -0.9 2.2 -0.9 z" fill="#ccf79a"/>'
        # trajetória
        '<path d="M38 34 Q62 4 92 17" fill="none" stroke="#27367a" stroke-width="1.4" '
        'stroke-dasharray="3 3" stroke-linecap="round"/>'
        # Terra
        '<circle cx="24" cy="46" r="18" fill="#4f6fd6"/>'
        '<path d="M12 40 Q16 32 25 34 Q29 38 24 42 Q22 47 16 46 Q11 45 12 40 Z" fill="#ccf79a"/>'
        '<path d="M28 52 Q34 49 38 54 Q37 60 31 61 Q27 58 28 52 Z" fill="#ccf79a"/>'
        '<path d="M8 52 Q11 50 13 54 Q11 58 8 56 Z" fill="#ccf79a"/>'
        '<circle cx="24" cy="46" r="18" fill="none" stroke="#c5d2fc" stroke-width="1.5"/>'
        # satélite
        '<g transform="translate(98,19) rotate(-25)">'
        '<rect x="-18" y="-3" width="11" height="6" rx="1" fill="#94aef7"/>'
        '<rect x="7" y="-3" width="11" height="6" rx="1" fill="#94aef7"/>'
        '<line x1="-7" y1="0" x2="-5" y2="0" stroke="#27367a" stroke-width="1"/>'
        '<line x1="5" y1="0" x2="7" y2="0" stroke="#27367a" stroke-width="1"/>'
        '<rect x="-5" y="-5" width="10" height="10" rx="2" fill="#27367a"/>'
        '<circle cx="0" cy="0" r="2" fill="#ccf79a"/>'
        '<line x1="0" y1="-5" x2="0" y2="-9" stroke="#27367a" stroke-width="1"/>'
        '<circle cx="0" cy="-9.5" r="1.2" fill="#27367a"/>'
        '</g>'
    )
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 70" width="120" height="70">'
           f'{parts}</svg>')
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")

def build_rocket_uri(flip=False):
    """Rocket as an SVG data-URI (points right). flip=True mirrors it to point left."""
    body = (
        # chama
        '<path d="M8 11.5 Q2 12 0 14 Q2 16 8 16.5 Z" fill="#ff9f1c" stroke="#27367a" stroke-width="0.8" stroke-linejoin="round"/>'
        '<path d="M8 12.6 Q4.5 13 3.5 14 Q4.5 15 8 15.4 Z" fill="#ffe066"/>'
        # aletas (atrás do corpo)
        '<polygon points="12,8 8,1 22,8" fill="#ff6b4a" stroke="#27367a" stroke-width="1" stroke-linejoin="round"/>'
        '<polygon points="12,20 8,27 22,20" fill="#ff6b4a" stroke="#27367a" stroke-width="1" stroke-linejoin="round"/>'
        # bocal
        '<rect x="7" y="11" width="5" height="6" rx="1.2" fill="#27367a"/>'
        # corpo
        '<path d="M12 8 H30 Q46 8 54 14 Q46 20 30 20 H12 Q8 14 12 8 Z" fill="#ffffff" stroke="#27367a" stroke-width="1.2" stroke-linejoin="round"/>'
        # risca
        '<rect x="26" y="8.6" width="3" height="10.8" fill="#27367a"/>'
        # cone do nariz
        '<path d="M37 8.8 Q47 9.8 53.4 14 Q47 18.2 37 19.2 Q40.5 14 37 8.8 Z" fill="#ff6b4a"/>'
        # janela
        '<circle cx="19" cy="14" r="3.6" fill="#94aef7" stroke="#27367a" stroke-width="1.2"/>'
        '<circle cx="18" cy="13" r="1" fill="#ffffff"/>'
    )
    if flip:
        body = f'<g transform="translate(56,0) scale(-1,1)">{body}</g>'
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 56 28" width="56" height="28">'
           f'{body}</svg>')
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")

def build_radar_uri(items):
    """4-axis radar as an SVG data-URI. items are in MEASURES order: top, right, bottom, left."""
    cx, cy, R = 320, 235, 140
    dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    lo, hi = 1 / 3, 2 / 3

    def pts(fracs):
        return " ".join(f"{cx + d[0] * R * f:.1f},{cy + d[1] * R * f:.1f}" for f, d in zip(fracs, dirs))

    def star(x, y, s, fill, stroke="none", sw=0):
        """4-point sparkle centred on (x, y), radius s."""
        k = s * 0.3
        p = [(x, y - s), (x + k, y - k), (x + s, y), (x + k, y + k),
             (x, y + s), (x - k, y + k), (x - s, y), (x - k, y - k)]
        coords = " ".join(f"{a:.1f},{b:.1f}" for a, b in p)
        return (f'<polygon points="{coords}" fill="{fill}" stroke="{stroke}" '
                f'stroke-width="{sw}" stroke-linejoin="round"/>')

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 445" width="640" height="445" '
           f'font-family="Helvetica, Arial, sans-serif">']

    # --- decorative background: orbit, sparkles, mini constellation ---
    svg.append(f'<circle cx="{cx}" cy="{cy}" r="{R * 1.10:.1f}" fill="none" stroke="#c5d2fc" '
               f'stroke-width="1" stroke-dasharray="2 5"/>')

    svg.append('<g opacity="0.5">')

    for sx, sy, s, color in ((60, 40, 6, "#8c9bd0"), (40, 380, 5, "#8c9bd0"), (600, 400, 6, "#8c9bd0"),
                             (110, 150, 4, "#8c9bd0"), (530, 300, 4, "#8c9bd0"), (250, 18, 4, "#8c9bd0"),
                             (70, 300, 4, "#9ccc65"), (580, 190, 5, "#9ccc65")):
        svg.append(star(sx, sy, s, color))

    cons = [(505, 28), (535, 52), (565, 40), (592, 72), (555, 94)]
    for a, b in ((0, 1), (1, 2), (2, 3), (1, 4)):
        svg.append(f'<line x1="{cons[a][0]}" y1="{cons[a][1]}" x2="{cons[b][0]}" y2="{cons[b][1]}" '
                   f'stroke="#8c9bd0" stroke-width="1.2"/>')
    for n, (sx, sy) in enumerate(cons):
        svg.append(star(sx, sy, 6 if n == 1 else 4.5, "#9ccc65" if n == 1 else "#8c9bd0"))
    svg.append('</g>')
    # --- radar ---
    svg.append(f'<polygon points="{pts([1] * 4)}" fill="none" stroke="#b4b2a9" stroke-width="1"/>')
    svg.append(f'<polygon points="{pts([hi] * 4)}" fill="#ccf79a" fill-opacity="0.65" stroke="#8fb85a" '
               f'stroke-width="1" stroke-dasharray="4 3"/>')
    svg.append(f'<polygon points="{pts([lo] * 4)}" fill="#ffffff" stroke="#8fb85a" '
               f'stroke-width="1" stroke-dasharray="4 3"/>')
    svg.append(f'<line x1="{cx}" y1="{cy - R}" x2="{cx}" y2="{cy + R}" stroke="#888780" stroke-width="1"/>')
    svg.append(f'<line x1="{cx - R}" y1="{cy}" x2="{cx + R}" y2="{cy}" stroke="#888780" stroke-width="1"/>')

    for txt, f in (("Alto", (1 + hi) / 2), ("Médio", 0.5), ("Baixo", lo / 2)):
        svg.append(f'<text x="{cx + 8}" y="{cy - R * f + 4:.1f}" font-size="11" fill="#555">{txt}</text>')

    svg.append(f'<polygon points="{pts([i["frac"] for i in items])}" fill="#94aef7" fill-opacity="0.55" '
               f'stroke="#4f6fd6" stroke-width="2.5" stroke-linejoin="round"/>')

    # vertices as stars (the participant's "constellation")
    for i, d in zip(items, dirs):
        x, y = cx + d[0] * R * i["frac"], cy + d[1] * R * i["frac"]
        svg.append(star(x, y, 11, "#27367a", "#ffffff", 1.2))
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.8" fill="#ccf79a"/>')

    # labels: penia name / measure / value · classification
    for i, d in zip(items, dirs):
        lines = [(i["penia"], 15, "700", "#111"), (i["short"], 12, "400", "#444"),
                 (f'{i["value_txt"]} · {i["label"]}', 12, "700", "#27367a")]
        if d == (0, -1):
            anchor, x, y0 = "middle", cx, cy - R - 52
        elif d == (0, 1):
            anchor, x, y0 = "middle", cx, cy + R + 26
        elif d == (1, 0):
            anchor, x, y0 = "start", cx + R + 18, cy - 8
        else:
            anchor, x, y0 = "end", cx - R - 18, cy - 8
        for n, (txt, size, weight, color) in enumerate(lines):
            svg.append(f'<text x="{x}" y="{y0 + n * 17}" text-anchor="{anchor}" font-size="{size}" '
                       f'font-weight="{weight}" fill="{color}">{escape(txt)}</text>')
    svg.append("</svg>")
    return "data:image/svg+xml;base64," + base64.b64encode("".join(svg).encode("utf-8")).decode("ascii")

# ---------------------------------------------------------------------
# Context for the template
# ---------------------------------------------------------------------
def build_context(row, refs, samples):
    """Build the dict of values that fills the HTML template for one participant."""
    g = lambda key: row[COL[key]]
    chrono_age = int(g("chrono_age"))
    bio_age = int(round(g("bio_age")))
    acceleration = bio_age - chrono_age
    group_label = str(g("age_group")).strip()
    sex = normalize_sex(g("sex"))
    ref_row = reference_row(refs, group_label, sex)       # only the participant's own sex
    age_group_name = GROUP_DISPLAY.get(group_label, group_label)

    # --- 4 main measures: value, reference, classification ---
    items = []
    for m in MEASURES:
        value = g(m["key"])
        mean = ref_row[REF_COL[m["key"]]]
        cls = classify(value, get_sample(samples, group_label, sex, m["key"]), m["key"])
        items.append(dict(
            **m, **cls,
            value_txt=f"{fmt_measure(m, value)} {m['unit']}",
            ref_txt=f"{fmt_fixed(mean, m['dec'])} {m['unit']}",
        ))

    measures = [dict(name=i["name"], meaning=i["meaning"], icon=icon_uri(i["icon"]),
                     value=i["value_txt"], ref=i["ref_txt"],
                     level=i["level"], label=i["label"], pos=i["pos"]) for i in items]
    bands = dict(low=33.33, mid=33.34, high=33.33)

    sex_pl = SEX_DISPLAY[sex].lower()
    legend = [
        dict(label="Baixo", text="Os seus valores encontram-se no terço mais baixo do grupo."),
        dict(label="Médio", text="Os seus valores encontram-se no terço intermédio do grupo."),
        dict(label="Alto", text="Os seus valores encontram-se no terço mais alto do grupo."),
    ]
    ref_note = (
        f"* O seu grupo corresponde aos participantes do sexo "
        f"{SEX_ADJ[sex]} da mesma faixa etária ({age_group_name}). "
        "A faixa etária foi definida com base na idade cronológica."
    )
    radar_intro = (
        "Ligados entre si, os quatro pontos dos seus resultados formam a sua própria <b>constelação</b>. "
        "Valores mais elevados tendem a refletir características musculares e ósseas mais favoráveis. "
        "A zona verde representa o intervalo médio de "
        f"{sex_pl} da sua faixa etária."
    )

    # --- Tanita scale ---
    def segments(prefix):
        return [
            dict(label="Braço direito", value=fmt(g(f"{prefix}_right_arm"))),
            dict(label="Braço esquerdo", value=fmt(g(f"{prefix}_left_arm"))),
            dict(label="Perna direita", value=fmt(g(f"{prefix}_right_leg"))),
            dict(label="Perna esquerda", value=fmt(g(f"{prefix}_left_leg"))),
            dict(label="Tronco", value=fmt(g(f"{prefix}_trunk"))),
        ]

    date_value = g("date")
    date_str = date_value if isinstance(date_value, str) else pd.to_datetime(date_value).strftime("%d/%m/%Y")

    info_know_more = ("Os resultados completos deste estudo farão parte da dissertação de mestrado da Universidade Nova de Lisboa, "
                      "intitulada <i>“Penia Map: Integrating Muscle-Based Biomarkers for Predicting Biological Age”</i>. Se tiver interesse "
                      f"em ser {info_know_more_sex[sex]} quando estiver disponível, ou se tiver qualquer questão sobre este relatório ou sobre o estudo, "
                      "contacte a equipa de investigação.")

    return dict(
        name=g("name"),
        date=date_str,
        logo_faculty=image_uri(LOGO_FACULTY),
        logo_lab=image_uri(LOGO_LAB),
        logo_project=image_uri(LOGO_PROJECT),
        space_uri=build_space_uri(),
        chrono_age=chrono_age,
        first_name=str(g("name")).strip().split()[0],
        welcome=welcome_sex[sex],
        rocket_uri=build_rocket_uri(flip=bio_age < chrono_age),
        bio_age=bio_age,
        acceleration=f"{acceleration:+d}".replace("-", "−"),
        acceleration_unit=years_pt(acceleration),
        info_know_more = info_know_more,
        tl=build_timeline(chrono_age, bio_age),
        sex_label=SEX_DISPLAY[sex],
        age_group_name=age_group_name,
        measures=measures,
        bands=bands,
        legend=legend,
        ref_note=ref_note,
        radar_intro=radar_intro,
        radar_uri=build_radar_uri(items),
        declines=[
            dict(name="Sarcopenia", desc="Perda de massa muscular", icon=icon_uri("sarcopenia")),
            dict(name="Dinapenia", desc="Perda de força muscular", icon=icon_uri("dynapenia")),
            dict(name="Osteopenia", desc="Perda de densidade mineral óssea", icon=icon_uri("osteopenia")),
            dict(name="Powerpenia", desc="Perda de potência muscular", icon=icon_uri("powerpenia")),
        ],
        anthropometrics=[
            dict(label="Altura", value=fmt(g("height")), unit="cm"),
            dict(label="Massa corporal", value=fmt(g("weight")), unit="kg"),
            dict(label="Índice de Massa corporal", value=fmt(g("bmi")), unit="kg/m<sup>2</sup>"),
        ],
        blocks=[
            dict(title="Massa muscular", total=fmt(g("mm_total")), unit="kg", segments=segments("mm")),
            dict(title="Gordura corporal", total=fmt(g("fat_total")), unit="%", segments=segments("fat")),
        ],
        others=[
            dict(label="Massa óssea", value=fmt(g("bone_mass")), unit="kg"),
            dict(label="Percentagem de água", value=fmt(g("water")), unit="%"),
            dict(label="Ingestão calórica diária sugerida", value=fmt(g("calories"), 0), unit="calorias"),
        ],

    )


def main():
    all_participants = pd.read_excel(EXCEL_FILE, sheet_name=PARTICIPANTS_SHEET)
    refs = load_reference(pd.read_excel(EXCEL_FILE, sheet_name=REFERENCE_SHEET))
    samples = build_samples(all_participants)         # always from the full sample

    participants = all_participants
    if TEST_PARTICIPANT is not None:
        names = participants[COL["name"]].astype(str).str.strip()
        participants = participants[names == str(TEST_PARTICIPANT)]
        if participants.empty:
            raise ValueError(
                f"Participant '{TEST_PARTICIPANT}' not found. "
                f"Examples of values in '{COL['name']}': {list(names.head(5))}"
            )

    env = Environment(loader=FileSystemLoader(TEMPLATE_FILE.parent), autoescape=True)
    template = env.get_template(TEMPLATE_FILE.name)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    errors = []
    for _, row in participants.iterrows():
        name = str(row[COL["name"]])
        try:
            html = template.render(**build_context(row, refs, samples))
            target = OUTPUT_DIR / f"Report_{name.replace(' ', '_')}.pdf"
            HTML(string=html, base_url=str(TEMPLATE_FILE.parent)).write_pdf(target)
            print(f"✔ {target}")
        except Exception as e:  # one bad row must not stop the whole batch
            errors.append((name, e))
            print(f"✘ {name}: {e!r}")

    if errors:
        print(f"\n{len(errors)} report(s) failed — check the data of those participants.")


if __name__ == "__main__":
    main()