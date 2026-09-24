
"""Plot train/test R2 vs k for each target."""

import glob
import os
import re

import matplotlib.pyplot as plt
import pandas as pd

# ---- Set your paths here ----
INPUT_DIR = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\results\sem outliers seed 42"
OUTPUT_DIR = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\results\sem outliers seed 42\plots"
# -----------------------------

# Read every file and add its k value as a column
frames = []
paths = glob.glob(os.path.join(INPUT_DIR, "resultados_test_k*_penia_map.xlsx")) + \
        glob.glob(os.path.join(INPUT_DIR, "resultados_test_k*_ant.xlsx"))

for path in paths:
    filename = os.path.basename(path)
    k = int(re.search(r"_k(\d+)_", filename).group(1))
    condition = "penia_map" if "penia_map" in filename else "ant"
    df = pd.read_excel(path)
    df["k"] = k
    df["condition"] = condition
    frames.append(df)

data = pd.concat(frames).sort_values("k")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Style for each condition/dataset combination
styles = {
    ("penia_map", "train"): dict(marker="o", linestyle="-", color="tab:blue", label="penia_map - Train"),
    ("penia_map", "test"): dict(marker="o", linestyle="--", color="tab:blue", label="penia_map - Test"),
    ("ant", "train"): dict(marker="s", linestyle="-", color="tab:orange", label="ant - Train"),
    ("ant", "test"): dict(marker="s", linestyle="--", color="tab:orange", label="ant - Test"),
}

# One plot per target, both conditions together
for target in data["target"].unique():
    sub = data[data["target"] == target]

    plt.figure(figsize=(8, 5))
    for (condition, dataset), style in styles.items():
        line = sub[(sub["condition"] == condition) & (sub["dataset"] == dataset)]
        if line.empty:
            continue
        plt.plot(line["k"], line["r2"], **style)

    plt.axhline(0, color="grey", linestyle=":", linewidth=0.8)
    plt.xticks(sorted(sub["k"].unique()))
    plt.xlabel("k")
    plt.ylabel("R²")
    plt.title(f"{target} — R² vs k")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    filename = re.sub(r"\W+", "_", target).strip("_")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{filename}.png"), dpi=150)
    plt.show()