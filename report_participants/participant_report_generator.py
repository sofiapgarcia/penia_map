"""
Functions to generate personalised PDF reports for each study participant (muscle, strength, bone and power
biomarkers, biological age and body composition), from an Excel sheet and an HTML template.

Available Functions
-------------------
[Public]
normalize_sex(value): Map a raw Gender cell to SEX_FEMALE / SEX_MALE.
build_samples(participants): Collect measure values per (age group, sex) and per sex.
get_sample(samples, group, sex, key): Get the reference sample for a measure.
classify(value, sample, key): Classify a value into a tercile (Baixo / Médio / Alto).
build_context(row, samples): Build the template context for one participant.
main(): Generate the PDF reports for all (or one test) participants.

[Private]
_fmt(x, decimals): Format a number without trailing zeros.
_fmt_fixed(x, decimals): Format a number with a fixed number of decimals.
_fmt_measure(m, x): Format a participant value for one of the 4 main measures.
_join_pt(items): Join items in Portuguese ("a, b e c").
_icon_uri(stem): File URI of an icon in ICONS_DIR.
_image_uri(path): File URI of an image path.
_sex_key(value): Normalise a Gender cell to a lookup key.
_years_pt(n): Portuguese singular/plural of "year".
_build_timeline(chrono, bio): Build the age timeline scale.
_build_space_uri(): Earth + trajectory + satellite SVG data-URI.
_build_rocket_uri(flip): Rocket SVG data-URI.
_build_radar_uri(items): 4-axis radar SVG data-URI.
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #

import math
import os
os.environ["WEASYPRINT_DLL_DIRECTORIES"] = r"C:\msys64\ucrt64\bin"
import base64
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML

# internal imports

# ------------------------------------------------------------------------------------------------------------------- #
# file specific constants
# ------------------------------------------------------------------------------------------------------------------- #

BASE = Path(r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos")
EXCEL_FILE = BASE / "report_participants_all.xlsx"
OUTPUT_DIR = BASE / "participant_reports"
ICONS_DIR = BASE / "participant_reports" / "imagens"   # sarcopenia.png, dynapenia.png, osteopenia.png, powerpenia.png
LOGO_PROJECT = ICONS_DIR / "logos.png"
TEMPLATE_FILE = BASE / "participant_reports" / "template_report_2.html"

# None -> generate for ALL participants.
# Otherwise: exactly the value in the "Name" column
TEST_PARTICIPANT = "Francisco Miguel Latas Figueira"

# Column names in the participants sheet (left = internal key, right = column name)
COL = {
    "name": "Name",
    "date": "Data",
    "sex": "Gender",
    "chrono_age": "Age (y)",
    "bio_age": "Estimated metabolic age",
    "age_group": "Age group (MeSH)",
    # main measures
    "muscle_mass": "Global muscle kg",
    "grip_strength": "ihd_values_max",
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
    "rel_muscle_mass": "rel_muscle_mass",
    "rel_grip": "rel_grip",
    "rel_peak_power": "rel_peak_power",
}

SEX_FEMALE, SEX_MALE = "Female", "Male"

# Accepted values in the "Gender" column (after _sex_key normalisation) -> SEX_FEMALE / SEX_MALE
SEX_MAP = {
    "0": SEX_FEMALE, "female": SEX_FEMALE, "f": SEX_FEMALE, "feminino": SEX_FEMALE, "mulher": SEX_FEMALE,
    "1": SEX_MALE,   "male": SEX_MALE,     "m": SEX_MALE, "masculino": SEX_MALE,   "homem": SEX_MALE,
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
         meaning="A rapidez com que consegue gerar força de preensão com a mão."),
]
LEVELS = {"low": "Abaixo da média", "mid": "Na média", "high": "Acima da média"}   # absolutos
LEVELS_REL = {"low": "Baixo", "mid": "Médio", "high": "Alto"}                      # relativos

# Relative measures (normalised by body mass). BMD is kept: it is already per area.
MEASURES_REL = [
    dict(key="rel_muscle_mass", penia="Sarcopenia", icon="sarcopenia", name="Índice de massa muscular",
         short="Massa muscular / altura²", det="o índice de massa muscular", unit="kg/m²", dec=1, trim=False,
         meaning="Massa muscular ajustada à altura"),
    dict(key="rel_grip", penia="Dinapenia", icon="dynapenia", name="Força de preensão relativa",
         short="Força / massa corporal", det="a força de preensão relativa", unit="kg/kg", dec=2, trim=False,
         meaning="Força de preensão ajustada à massa corporal"),
    MEASURES[2],   # bmd (unchanged)
    dict(key="rel_peak_power", penia="Powerpenia", icon="powerpenia", name="Pico de potência relativo",
         short="Potência / massa corporal", det="o pico de potência relativo", unit="W/kg", dec=2, trim=False,
         meaning="Pico de potência ajustado à massa corporal"),
]

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #
def add_relative_columns(df):
    """
    Add the normalised columns used by the relative radar: muscle mass / height² (kg/m²),
    grip strength and peak power / body mass.

    :param df: DataFrame with all participants
    :return: Copy of df with the relative columns added
    """
    df = df.copy()
    num = lambda k: pd.to_numeric(df[COL[k]], errors="coerce")
    h2 = (num("height") / 100) ** 2
    w = num("weight")
    df[COL["rel_muscle_mass"]] = num("muscle_mass") / h2
    df[COL["rel_grip"]] = num("grip_strength") / w
    df[COL["rel_peak_power"]] = num("peak_power") / w
    return df

def normalize_sex(value):
    """
    Convert a raw cell of the Gender column to SEX_FEMALE or SEX_MALE.

    :param value: Raw value of the Gender column
    :return: SEX_FEMALE or SEX_MALE
    :raises ValueError: If the value is not present in SEX_MAP
    """
    key = _sex_key(value)
    if key in SEX_MAP:
        return SEX_MAP[key]
    raise ValueError(f"Unrecognised value '{value}' in column '{COL['sex']}' - add it to SEX_MAP.")


def build_samples(participants):
    """
    Collect the values of every measure per (age group, sex) and per sex. Uses ALL participants (before filtering).

    :param participants: DataFrame with all participants
    :return: Tuple (by_group, by_sex) of dicts mapping (group, sex, key) / (sex, key) to numpy arrays
    """
    df = participants.copy()
    df["_group"] = df[COL["age_group"]].astype(str).str.strip()
    df["_sex"] = df[COL["sex"]].map(lambda v: SEX_MAP.get(_sex_key(v)))
    by_group, by_sex = {}, {}
    for m in MEASURES + MEASURES_REL:
        df["_v"] = pd.to_numeric(df[COL[m["key"]]], errors="coerce")
        for (grp, sex), s in df.groupby(["_group", "_sex"])["_v"]:
            s = s.dropna()
            if len(s) >= MIN_N_FOR_TERCILES:
                by_group[(grp, sex, m["key"])] = s.to_numpy()
        for sex, s in df.groupby("_sex")["_v"]:
            by_sex[(sex, m["key"])] = s.dropna().to_numpy()
    return by_group, by_sex


def get_sample(samples, group, sex, key):
    """
    Get the reference sample of a measure, falling back to the whole sex when the (group, sex) cell is too small.

    :param samples: Tuple (by_group, by_sex) returned by build_samples
    :param group: Age group label (MeSH)
    :param sex: SEX_FEMALE or SEX_MALE
    :param key: Internal measure key (e.g. "muscle_mass")
    :return: Numpy array with the reference values
    :raises ValueError: If fewer than 3 values are available
    """
    by_group, by_sex = samples
    arr = by_group.get((group, sex, key))
    if arr is None:
        arr = by_sex.get((sex, key))
    if arr is None or len(arr) < 3:
        raise ValueError(f"Not enough data to compute terciles for '{key}' ({sex}).")
    return arr


def classify(value, sample, key, levels=LEVELS):
    """
    Compute the percentile of a value within its reference sample and map it to a tercile (Baixo / Médio / Alto).

    :param value: Participant value
    :param sample: Numpy array with the reference values
    :param key: Internal measure key (used in error messages)
    :return: Dict with pct, level, label, frac and pos
    :raises ValueError: If the value is missing
    """
    if pd.isna(value):
        raise ValueError(f"Missing value for '{key}'.")
    v = float(value)
    pct = ((sample < v).sum() + 0.5 * (sample == v).sum()) / len(sample)    # 0..1
    level = "low" if pct < 1 / 3 else "high" if pct > 2 / 3 else "mid"
    return dict(pct=pct, level=level, label=levels[level], frac=pct, pos=round(pct * 100, 1))


def build_context(row, samples):
    """
    Build the dict of values that fills the HTML template for one participant.

    :param row: Row (Series) of the participants DataFrame
    :param samples: Tuple (by_group, by_sex) returned by build_samples
    :return: Dict with all template variables
    """
    g = lambda key: row[COL[key]]
    chrono_age = int(g("chrono_age"))
    bio_age = int(round(g("bio_age")))
    acceleration = bio_age - chrono_age
    group_label = str(g("age_group")).strip()
    sex = normalize_sex(g("sex"))
    age_group_name = GROUP_DISPLAY.get(group_label, group_label)

    # --- 4 main measures: value, reference, classification ---
    items = _build_items(MEASURES, g, group_label, sex, samples, LEVELS)
    items_rel = _build_items(MEASURES_REL, g, group_label, sex, samples, LEVELS_REL)

    measures = [dict(name=i["name"], meaning=i["meaning"], icon=_icon_uri(i["icon"]),
                     value=i["value_txt"], ref=i["ref_txt"],
                     level=i["level"], label=i["label"], pos=i["pos"]) for i in items]
    bands = dict(low=33.33, mid=33.34, high=33.33)

    sex_pl = SEX_DISPLAY[sex].lower()
    legend = [
        dict(label=LEVELS["low"], text="Os seus valores encontram-se no terço mais baixo do grupo."),
        dict(label=LEVELS["mid"], text="Os seus valores encontram-se no terço intermédio do grupo."),
        dict(label=LEVELS["high"], text="Os seus valores encontram-se no terço mais alto do grupo."),
    ]
    ref_note = (
        f"* O seu grupo corresponde aos participantes do sexo "
        f"{SEX_ADJ[sex]} da mesma faixa etária ({age_group_name}). "
        "A faixa etária foi definida com base na idade cronológica, "
        "segundo os critérios de classificação etária do MeSH "
        "(Medical Subject Headings)."
    )
    radar_intro = (
        "Ligados entre si, os quatro pontos dos seus resultados formam a sua própria <b>constelação</b>. "
        "Estes são valores absolutos, ou seja, não têm em conta o tamanho do corpo. "
        "Valores mais elevados tendem a refletir características musculares e ósseas mais favoráveis. "
        "A zona cinzenta representa o intervalo médio de "
        f"{sex_pl} da sua faixa etária."
    )
    radar_rel_intro = (
        "Para uma comparação mais justa entre pessoas de tamanhos diferentes, este segundo mapa ajusta os "
        "resultados ao tamanho do corpo: a massa muscular é dividida pela altura ao quadrado, e a força de "
        "preensão e o pico de potência pela massa corporal. A densidade mineral óssea "
        "mantém-se igual, por já ser expressa por área. A zona cinzenta representa o intervalo médio de "
        f"{sex_pl} da sua faixa etária."
    )

    # --- Tanita scale ---
    def segments(prefix):
        """
        Build the list of body segments (arms, legs, trunk) for a given measure prefix.

        :param prefix: Column key prefix ("mm" for muscle mass, "fat" for body fat)
        :return: List of dicts with label and formatted value
        """
        return [
            dict(label="Braço direito", value=_fmt(g(f"{prefix}_right_arm"))),
            dict(label="Braço esquerdo", value=_fmt(g(f"{prefix}_left_arm"))),
            dict(label="Perna direita", value=_fmt(g(f"{prefix}_right_leg"))),
            dict(label="Perna esquerda", value=_fmt(g(f"{prefix}_left_leg"))),
            dict(label="Tronco", value=_fmt(g(f"{prefix}_trunk"))),
        ]

    date_value = g("date")
    date_str = date_value if isinstance(date_value, str) else pd.to_datetime(date_value).strftime("%d/%m/%Y")

    info_know_more = ("Os resultados completos deste estudo farão parte da dissertação de mestrado da Universidade Nova de Lisboa, "
                      "intitulada <i>“Penia Map: Integrating Muscle-Based Biomarkers for Predicting Biological Age”</i>. Se tiver interesse "
                      f"em ser {info_know_more_sex[sex]} quando estiver disponível, ou se tiver qualquer questão sobre este relatório ou sobre o estudo, "
                      "contacte sp.garcia@campus.fct.unl.pt.")

    return dict(
        name=g("name"),
        date=date_str,
        logo_project=_image_uri(LOGO_PROJECT),
        space_uri=_build_space_uri(),
        chrono_age=chrono_age,
        first_name=str(g("name")).strip().split()[0],
        welcome=welcome_sex[sex],
        rocket_uri=_build_rocket_uri(flip=bio_age < chrono_age),
        bio_age=bio_age,
        acceleration=f"{acceleration:+d}".replace("-", "−"),
        acceleration_unit=_years_pt(acceleration),
        info_know_more = info_know_more,
        tl=_build_timeline(chrono_age, bio_age),
        sex_label=SEX_DISPLAY[sex],
        age_group_name=age_group_name,
        measures=measures,
        bands=bands,
        legend=legend,
        ref_note=ref_note,
        radar_intro=radar_intro,
        level_labels=[LEVELS["low"], LEVELS["mid"], LEVELS["high"]],
        radar_rel_intro=radar_rel_intro,
        radar_uri=_build_radar_uri(items, axis=("Acima", "Na média", "Abaixo"), split_label=True),
        radar_rel_uri=_build_radar_uri(items_rel),
        declines=[
            dict(name="Sarcopenia", desc="Perda de massa muscular", icon=_icon_uri("sarcopenia")),
            dict(name="Dinapenia", desc="Perda de força muscular", icon=_icon_uri("dynapenia")),
            dict(name="Osteopenia", desc="Perda de densidade mineral óssea", icon=_icon_uri("osteopenia")),
            dict(name="Powerpenia", desc="Perda de potência muscular", icon=_icon_uri("powerpenia")),
        ],
        anthropometrics=[
            dict(label="Altura", value=_fmt(g("height")), unit="cm"),
            dict(label="Massa corporal", value=_fmt(g("weight")), unit="kg"),
            dict(label="Índice de Massa corporal", value=_fmt(g("bmi")), unit="kg/m<sup>2</sup>"),
        ],
        blocks=[
            dict(title="Massa muscular", total=_fmt(g("mm_total")), unit="kg", segments=segments("mm")),
            dict(title="Gordura corporal", total=_fmt(g("fat_total")), unit="%", segments=segments("fat")),
        ],
        others=[
            dict(label="Massa óssea", value=_fmt(g("bone_mass")), unit="kg"),
            dict(label="Percentagem de água", value=_fmt(g("water")), unit="%"),
            dict(label="Ingestão calórica diária sugerida", value=_fmt(g("calories"), 0), unit="calorias"),
        ],

    )


def main():
    """
    Read the Excel file, render the HTML template for each participant and write one PDF report per participant.

    :return: None
    """
    all_participants = add_relative_columns(pd.read_excel(EXCEL_FILE))
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
            html = template.render(**build_context(row, samples))
            target = OUTPUT_DIR / f"Report_{name.replace(' ', '_')}.pdf"
            HTML(string=html, base_url=str(TEMPLATE_FILE.parent)).write_pdf(target)
            print(f"✔ {target}")
        except Exception as e:  # one bad row must not stop the whole batch
            errors.append((name, e))
            print(f"✘ {name}: {e!r}")

    if errors:
        print(f"\n{len(errors)} report(s) failed — check the data of those participants.")


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #


def _fmt(x, decimals=1):
    """
    Format a number without useless zeros (41.40 -> 41.4; 7.0 -> 7).

    :param x: Number to format
    :param decimals: Maximum number of decimals
    :return: Formatted string, or "—" if the value is missing
    """
    if pd.isna(x):
        return "—"
    s = f"{float(x):.{decimals}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s.replace(".", ",") if DECIMAL_COMMA else s


def _fmt_fixed(x, decimals):
    """
    Format a number with a fixed number of decimals (e.g. 1.147, 14.50).

    :param x: Number to format
    :param decimals: Exact number of decimals
    :return: Formatted string, or "—" if the value is missing
    """
    if pd.isna(x):
        return "—"
    s = f"{float(x):.{decimals}f}"
    return s.replace(".", ",") if DECIMAL_COMMA else s


def _fmt_measure(m, x):
    """
    Format a participant value for one of the 4 main measures (no unit).

    :param m: Measure definition (element of MEASURES)
    :param x: Participant value
    :return: Formatted string
    """
    return _fmt(x, m["dec"]) if m["trim"] else _fmt_fixed(x, m["dec"])


def _join_pt(items):
    """
    Join items in Portuguese style ("a, b e c").

    :param items: Iterable of strings
    :return: Joined string
    """
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " e " + items[-1]


def _icon_uri(stem):
    """
    Get the file URI of an icon stored in ICONS_DIR.

    :param stem: Icon file name without extension (e.g. "sarcopenia")
    :return: File URI, or an empty string if the file does not exist
    """
    path = ICONS_DIR / f"{stem}.png"
    return path.resolve().as_uri() if path.exists() else ""


def _image_uri(path):
    """
    Return a file URI for an image path.

    :param path: Path to the image
    :return: File URI
    :raises FileNotFoundError: If the image does not exist
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    return path.resolve().as_uri()


def _sex_key(value):
    """
    Normalise a cell of the Gender column to a lookup key (1.0 -> '1', ' Male ' -> 'male').

    :param value: Raw value of the Gender column
    :return: Normalised string key, or "" if the value is missing
    """
    if pd.isna(value):
        return ""
    try:
        f = float(value)
        if f.is_integer():
            return str(int(f))
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower()


def _years_pt(n):
    """
    Return the Portuguese word for "year(s)": 'ano' for ±1, 'anos' for everything else (including 0).

    :param n: Number of years
    :return: "ano" or "anos"
    """
    return "ano" if abs(int(n)) == 1 else "anos"


def _build_timeline(chrono, bio):
    """
    Build a scale centred on the chronological age (always at 50%), with one tick per year.

    :param chrono: Chronological age
    :param bio: Biological (estimated metabolic) age
    :return: Dict with chrono and bio positions, bar left/width and the list of ticks
    """
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

def _build_items(measures, g, group_label, sex, samples, levels=LEVELS):
    """
    Value, reference mean and tercile classification for a list of measures.

    :param measures: List of measure definitions (MEASURES or MEASURES_REL)
    :param g: Function key -> row value
    :param group_label: Age group label (MeSH)
    :param sex: SEX_FEMALE or SEX_MALE
    :param samples: Tuple (by_group, by_sex) returned by build_samples
    :return: List of dicts (measure info + classification + formatted texts)
    """
    items = []
    for m in measures:
        value = g(m["key"])
        sample = get_sample(samples, group_label, sex, m["key"])
        cls = classify(value, sample, m["key"], levels)
        items.append(dict(
            **m, **cls,
            value_txt=f"{_fmt_measure(m, value)} {m['unit']}",
            ref_txt=f"{_fmt_fixed(sample.mean(), m['dec'])} {m['unit']}",
        ))
    return items

def _build_space_uri():
    """
    Build the Earth + trajectory + satellite illustration as an SVG data-URI.

    :return: SVG data-URI string
    """
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


def _build_rocket_uri(flip=False):
    """
    Build a rocket as an SVG data-URI (points right by default).

    :param flip: If True, mirror the rocket so it points left
    :return: SVG data-URI string
    """
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


def _build_radar_uri(items, axis=("Alto", "Médio", "Baixo"), split_label=False):
    """
    Build the 4-axis radar chart as an SVG data-URI. Items are in MEASURES order: top, right, bottom, left.

    :param items: List of dicts with the measure info and classification (frac, penia, short, value_txt, label)
    :return: SVG data-URI string
    """
    cx, cy, R = 320, 235, 140
    dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    lo, hi = 1 / 3, 2 / 3

    def pts(fracs):
        """
        Convert the fraction of each axis into SVG polygon points.

        :param fracs: Fractions (0..1) of the radius for each axis
        :return: String of "x,y" points
        """
        return " ".join(f"{cx + d[0] * R * f:.1f},{cy + d[1] * R * f:.1f}" for f, d in zip(fracs, dirs))

    def star(x, y, s, fill, stroke="none", sw=0):
        """
        Build a 4-point sparkle centred on (x, y).

        :param x: Centre x coordinate
        :param y: Centre y coordinate
        :param s: Radius of the sparkle
        :param fill: Fill colour
        :param stroke: Stroke colour
        :param sw: Stroke width
        :return: SVG polygon element as a string
        """
        k = s * 0.3
        p = [(x, y - s), (x + k, y - k), (x + s, y), (x + k, y + k),
             (x, y + s), (x - k, y + k), (x - s, y), (x - k, y - k)]
        coords = " ".join(f"{a:.1f},{b:.1f}" for a, b in p)
        return (f'<polygon points="{coords}" fill="{fill}" stroke="{stroke}" '
                f'stroke-width="{sw}" stroke-linejoin="round"/>')

    H = 445 + (17 if split_label else 0)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 {H}" width="640" height="{H}" '
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
    svg.append(f'<polygon points="{pts([hi] * 4)}" fill="#e3e6ee" stroke="#aab1c4" '
               f'stroke-width="1" stroke-dasharray="4 3"/>')
    svg.append(f'<polygon points="{pts([lo] * 4)}" fill="#ffffff" stroke="#aab1c4" '
               f'stroke-width="1" stroke-dasharray="4 3"/>')
    svg.append(f'<line x1="{cx}" y1="{cy - R}" x2="{cx}" y2="{cy + R}" stroke="#888780" stroke-width="1"/>')
    svg.append(f'<line x1="{cx - R}" y1="{cy}" x2="{cx + R}" y2="{cy}" stroke="#888780" stroke-width="1"/>')

    for txt, f in zip(axis, ((1 + hi) / 2, 0.5, lo / 2)):
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
        lines = [(i["penia"], 15, "700", "#111"), (i["short"], 12, "400", "#444")]
        if split_label:
            lines += [(i["value_txt"], 12, "700", "#27367a"), (i["label"], 12, "700", "#27367a")]
        else:
            lines += [(f'{i["value_txt"]} · {i["label"]}', 12, "700", "#27367a")]
        if d == (0, -1):
            anchor, x, y0 = "middle", cx, cy - R - 17 * len(lines) - 1
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


# ------------------------------------------------------------------------------------------------------------------- #
# entry point
# ------------------------------------------------------------------------------------------------------------------- #

if __name__ == "__main__":
    main()