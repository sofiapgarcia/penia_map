from constants import PENIA_MAP_GROUPS

import pandas as pd


# ============================================================
# Configuration
# ============================================================

FILE_PATH = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\final_dataframe.xlsx"

#OUTPUT_PATH = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\penia_map_pca_bmd.xlsx"
#OUTPUT_PATH = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\penia_map_pca_peak_power.xlsx"
OUTPUT_PATH = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\penia_map_pca.xlsx"

#VARIABLES_TO_REMOVE = ["Bone Mineral Density","Estimated bone mass kg"]
#VARIABLES_TO_REMOVE = ["peak_power","velocity_at_peak_power", "force_at_peak_power", "peak_power_mass", "peak_power_height_squared", "max_force", "max_velocity","rep_pp_hd_peak_rfd","mean_hd_peak_rfd"]
VARIABLES_TO_REMOVE = []
TOP_N_LOADINGS = 10


# ============================================================
# Read data
# ============================================================

df = pd.read_excel(FILE_PATH)


# ============================================================
# Apply PCA separately for each PENIA MAP domain
# ============================================================

pca_components = {}

for domain, variables in PENIA_MAP_GROUPS.items():

    # Variables that exist in the dataframe
    available_variables = [
        variable
        for variable in variables
        if variable in df.columns
    ]

    # Variables actually used in PCA
    used_variables = [
        variable
        for variable in available_variables
        if variable not in VARIABLES_TO_REMOVE
    ]

    # Variables not used
    not_used_variables = [
        variable
        for variable in variables
        if variable not in used_variables
    ]

    # --------------------------------------------------------
    # Skip if no variables remain
    # --------------------------------------------------------

    if not used_variables:
        print(f"\n{domain}: No variables available for PCA. Skipping.")
        continue

    # --------------------------------------------------------
    # Apply PCA
    # --------------------------------------------------------

    domain_df = df[used_variables].copy()

    pca, domain_pca, loadings = apply_pca(
        domain_df,
        n_components=1
    )

    pca_components[domain] = domain_pca["PC1"]

    # --------------------------------------------------------
    # Print explained variance
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print(domain)
    print("=" * 80)

    print(
        f"PC1 explained variance: "
        f"{pca.explained_variance_ratio_[0]:.2%}"
    )

    # --------------------------------------------------------
    # Get loadings for PC1
    # --------------------------------------------------------

    pc1_loadings = loadings["PC1"]

    # Sort by absolute loading
    pc1_loadings = pc1_loadings.reindex(
        pc1_loadings.abs()
        .sort_values(ascending=False)
        .index
    )

    # --------------------------------------------------------
    # Print top variables
    # --------------------------------------------------------

    print(f"\nTop {TOP_N_LOADINGS} variables contributing to PC1:")

    for i, (variable, loading) in enumerate(
        pc1_loadings.head(TOP_N_LOADINGS).items(),
        start=1
    ):
        print(
            f"{i}. {variable}: "
            f"loading = {loading:.4f}"
        )


# ============================================================
# Create final dataframe
# ============================================================

#final_pca_df = pd.DataFrame({"Bone Mineral Density": df["Bone Mineral Density"]}, index=df.index)
#final_pca_df = pd.DataFrame({"peak_power": df["peak_power"]}, index=df.index)
final_pca_df = pd.DataFrame({"Estimated metabolic age": df["Estimated metabolic age"],"Age (y)":df["Age (y)"]}, index=df.index)


# Add PC1 from each domain
for domain, pc1 in pca_components.items():
    final_pca_df[domain] = pc1


# ============================================================
# Save results
# ============================================================

final_pca_df.to_excel(OUTPUT_PATH, index=False)


print("\nPCA completed.")
print(f"Output saved to: {OUTPUT_PATH}")
print(f"Final dataframe shape: {final_pca_df.shape}")