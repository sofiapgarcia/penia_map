import pandas as pd

# Path to the Excel file
excel_path = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\final_dataframe.xlsx"

df = pd.read_excel(excel_path)

# ---------------------------------------------------------
# Basic sample information
# ---------------------------------------------------------

n_women = (df["Gender"] == 2).sum()
n_men = (df["Gender"] == 1).sum()
n_total = len(df)

# Age range
age_min = df["Age (y)"].min()
age_max = df["Age (y)"].max()


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def mean_sd(series):
    """Return mean ± standard deviation."""
    return f"{series.mean():.1f} ± {series.std():.1f}"


def activity_distribution(series):
    """Return n (%) for each physical activity level."""
    counts = series.value_counts()
    total = series.notna().sum()

    if total == 0:
        return "N/A"

    return (
        f"Low: {counts.get(1, 0)} "
        f"({counts.get(1, 0) / total * 100:.1f}%), "
        f"Moderate: {counts.get(2, 0)} "
        f"({counts.get(2, 0) / total * 100:.1f}%), "
        f"High: {counts.get(3, 0)} "
        f"({counts.get(3, 0) / total * 100:.1f}%)"
    )


# ---------------------------------------------------------
# Groups
# ---------------------------------------------------------

groups = {
    "Female": df[df["Gender"] == 2],
    "Male": df[df["Gender"] == 1],
    "Total": df
}


# ---------------------------------------------------------
# Create summary table
# ---------------------------------------------------------

table = pd.DataFrame(
    index=[
        "Age (years)",
        "Weight (kg)",
        "Height (cm)",
        "BMI (kg/m²)",
        "Physical activity level"
    ],
    columns=[
        "Female",
        "Male",
        "Total"
    ]
)

for group_name, group in groups.items():

    table.loc["Age (years)", group_name] = mean_sd(
        group["Age (y)"]
    )

    table.loc["Weight (kg)", group_name] = mean_sd(
        group["Body mass(kg)"]
    )

    table.loc["Height (cm)", group_name] = mean_sd(
        group["Height(cm)"]
    )

    table.loc["BMI (kg/m²)", group_name] = mean_sd(
        group["Body mass index (BMI)"]
    )

    table.loc["Physical activity level", group_name] = activity_distribution(
        group["Activity level"]
    )


# ---------------------------------------------------------
# Print results
# ---------------------------------------------------------

print("=" * 50)
print("RESULTS")
print("=" * 50)

print(f"Number of participants: {n_total}")
print(f"Number of women: {n_women}")
print(f"Number of men: {n_men}")
print(f"Age range: {age_min}–{age_max} years")

print("\nParticipant Characteristics")
print("=" * 80)
print(table.to_string())