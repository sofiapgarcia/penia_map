import re, shutil
import pandas as pd
from pathlib import Path

SOURCE = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos"
OUTPUT = r"C:\Users\Sofia\Desktop\dataset_peniamap"
IDS_EXCEL = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\features_power_dyna.xlsx"   # needs a column named ID

SKIP = {"onsets_validated", "sync_indices", "trim_indices"}
is_skipped = lambda n: Path(n).suffix.lower() in (".xlsx", ".xls") and (Path(n).stem in SKIP or re.fullmatch(r"S\d+_files_validation", Path(n).stem))
skip_excels = lambda folder, names: [n for n in names if is_skipped(n)]

ids = pd.read_excel(IDS_EXCEL)["subject"].dropna().astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
wanted = {"S" + re.sub(r"^[Ss]", "", i) for i in ids}

for folder in Path(SOURCE).iterdir():
    if folder.is_dir() and folder.name in wanted:
        shutil.copytree(folder, Path(OUTPUT) / folder.name, ignore=skip_excels, dirs_exist_ok=True)