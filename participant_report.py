"""
Generate one PDF report per participant from an Excel workbook with two sheets:
  - "Classified data"      -> one row per participant
  - "Means group x sex"    -> reference means per age group (MeSH) and sex

Dependencies: pip install pandas jinja2 weasyprint openpyxl
"""
import os
os.environ["WEASYPRINT_DLL_DIRECTORIES"] = r"C:\msys64\ucrt64\bin"
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML

# =====================================================================
# 1) CONFIGURATION
# =====================================================================
BASE = Path(r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\results")

EXCEL_FILE = BASE / "means_report_participant.xlsx"
PARTICIPANTS_SHEET = "Classified data"
REFERENCE_SHEET = "Means group x sex"
OUTPUT_DIR = BASE / "participant_reports"
ICONS_DIR = BASE / "participant_reports" / "imagens"   # sarcopenia.png, dynapenia.png, osteopenia.png, powerpenia.png
TEMPLATE_FILE = BASE / "participant_reports" / "template_report.html"

# None -> generate for ALL participants.
# Otherwise: exactly the value in the "Name" column (e.g. "Sujeito 125" or "125").
TEST_PARTICIPANT = "Íris Cristina Pólvora Piçarra"

# Column names in the participants sheet (left = internal key, right = YOUR column name)
COL = {
    "name": "Name",
    "date": "Date",
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

# Text shown in the report's table header for each MeSH age group
GROUP_DISPLAY = {
    "Young Adult (18-24)": "Jovem Adulto: 18-24 anos",
    "Adult (25-44)": "Adulto: 25-44 anos",
    "Middle Aged (45-64)": "Meia-idade: 45-64 anos",
    "Aged (>=65)": "Idoso: ≥65 anos",
}

DECIMAL_COMMA = False  # True -> 41,4 kg ; False -> 41.4 kg
# =====================================================================


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


def build_context(row, refs):
    """Build the dict of values that fills the HTML template for one participant."""
    g = lambda key: row[COL[key]]
    chrono_age = int(g("chrono_age"))
    bio_age = int(round(g("bio_age")))
    acceleration = bio_age - chrono_age
    group_label = str(g("age_group")).strip()
    ref_f_row = reference_row(refs, group_label, SEX_FEMALE)
    ref_m_row = reference_row(refs, group_label, SEX_MALE)

    def ref(measure, decimals):
        col = REF_COL[measure]
        return fmt_fixed(ref_f_row[col], decimals), fmt_fixed(ref_m_row[col], decimals)

    mm_f, mm_m = ref("muscle_mass", 1)
    gs_f, gs_m = ref("grip_strength", 1)
    bmd_f, bmd_m = ref("bmd", 3)
    pw_f, pw_m = ref("peak_power", 2)

    measures = [
        dict(name="Massa muscular", value=f"{fmt(g('muscle_mass'))} kg",
             ref_f=f"{mm_f} kg", ref_m=f"{mm_m} kg",
             meaning="Quantidade de músculo no corpo"),
        dict(name="Força de Preensão Isométrica", value=f"{fmt(g('grip_strength'))} kg",
             ref_f=f"{gs_f} kg", ref_m=f"{gs_m} kg",
             meaning="A força máxima que conseguiu exercer ao apertar o dinamómetro"),
        dict(name="Densidade Mineral Óssea (estimada)", value=f"{fmt_fixed(g('bmd'), 3)} g/cm²",
             ref_f=f"{bmd_f} g/cm²", ref_m=f"{bmd_m} g/cm²",
             meaning="Reflete a robustez dos ossos"),
        dict(name="Pico de potência", value=f"{fmt_fixed(g('peak_power'), 2)} W",
             ref_f=f"{pw_f} W", ref_m=f"{pw_m} W",
             meaning="A rapidez com que consegue gerar força com a mão."),
    ]

    def segments(prefix):
        return [
            dict(label="Braço direito", value=fmt(g(f"{prefix}_right_arm"))),
            dict(label="Braço esquerdo", value=fmt(g(f"{prefix}_left_arm"))),
            dict(label="Perna direita", value=fmt(g(f"{prefix}_right_leg"))),
            dict(label="Perna esquerda", value=fmt(g(f"{prefix}_left_leg"))),
            dict(label="Tronco", value=fmt(g(f"{prefix}_trunk"))),
        ]

    def icon(file_stem):
        path = ICONS_DIR / f"{file_stem}.png"
        return path.resolve().as_uri() if path.exists() else ""

    date_value = g("date")
    date_str = date_value if isinstance(date_value, str) else pd.to_datetime(date_value).strftime("%d/%m/%Y")

    return dict(
        name=g("name"),
        date=date_str,
        chrono_age=chrono_age,
        bio_age=bio_age,
        acceleration=f"{acceleration:+d}".replace("-", "−"),
        age_group_name=GROUP_DISPLAY.get(group_label, group_label),
        measures=measures,
        declines=[
            dict(name="Sarcopenia", desc="Perda de massa muscular", icon=icon("sarcopenia")),
            dict(name="Dinapenia", desc="Perda de força muscular", icon=icon("dynapenia")),
            dict(name="Osteopenia", desc="Perda de densidade mineral óssea", icon=icon("osteopenia")),
            dict(name="Powerpenia", desc="Perda de potência muscular", icon=icon("powerpenia")),
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
    participants = pd.read_excel(EXCEL_FILE, sheet_name=PARTICIPANTS_SHEET)
    refs = load_reference(pd.read_excel(EXCEL_FILE, sheet_name=REFERENCE_SHEET))

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
            html = template.render(**build_context(row, refs))
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