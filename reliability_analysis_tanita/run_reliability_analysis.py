"""
Functions to analyze agreement between DXA and TANITA measurements and characterize the study sample.

Available Functions
-------------------
[Public]
generate_sample_characterization_table(...): Generate a LaTeX table summarizing sample characteristics.

"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import pandas as pd

# internal imports
from reliability_analysis_tanita.ICC import compute_icc_pairs
from reliability_analysis_tanita.bland_altman import plot_bland_altman_pairs
# ------------------------------------------------------------------------------------------------------------------- #
# file specific constants
# ------------------------------------------------------------------------------------------------------------------- #
VARIABLE_MAPPING = {
    # DXA variable -> TANITA variable
    "Massa Corporal": "Body mass(kg)",
    "BMI_DXA": "Body mass index (BMI)",
    "Perc_gordura_total": "Global fat %",
    "Perc_gordura_braco_direito": "Arm fat (right) %",
    "Perc_gordura_braco_esquerdo": "Arm fat (left) %",
    "Perc_gordura_perna_direita": "Leg fat (right) %",
    "Perc_gordura_perna_esquerda": "Leg fat (left) %",
    "Perc_gordura_tronco": "Torso fat %",
    "Magro_total": "Global muscle kg",
    "Massa_magra_braco_direito": "Arm muscle (right) kg",
    "Massa_magra_braco_esquerdo": "Arm muscle (left) kg",
    "Massa_magra_perna_direita": "Leg muscle (right) kg",
    "Massa_magra_perna_esquerda": "Leg muscle (left) kg",
    "Magro_tronco": "Torso muscle kg",
    "BMC_total": "Estimated bone mass kg",
    "Altura": "Altura_testes"
}
# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def generate_sample_characterization_table(df,
                                                  age_col="Age", gender_col="Gender",
                                                  bmi_col="Body mass index (BMI)",
                                                  caption="Sample characteristics by sex and for the total sample.",
                                                  label="tab:sample_characterization"):
    """
    Generate a LaTeX table (Female/Male/Total columns) summarizing age, gender and BMI to participant group of icc analysis.

    :param df: DataFrame containing age, gender (1=male, 2=female), and BMI columns.
    :param age_col: Name of the age column.
    :param gender_col: Name of the gender column.
    :param bmi_col: Name of the BMI column.
    :param caption: Table caption.
    :param label: Table label.
    :return: None. Print a LateX table.
    """
    missing = [c for c in (age_col, gender_col, bmi_col) if c not in df.columns]
    if missing:
        raise KeyError(f"Missing column(s) {missing}. Available: {list(df.columns)}")

    female = df[df[gender_col] == 2]
    male = df[df[gender_col] == 1]

    n_female, n_male, n_total = len(female), len(male), len(df)

    def mean_sd(series):
        return f"{series.mean():.1f} $\\pm$ {series.std():.1f}"

    row_age = (mean_sd(female[age_col]), mean_sd(male[age_col]), mean_sd(df[age_col]))
    row_bmi = (mean_sd(female[bmi_col]), mean_sd(male[bmi_col]), mean_sd(df[bmi_col]))

    latex_table = r"""\begin{table}[H]
	\centering
	\caption{""" + caption + r"""}
	\label{""" + label + r"""}
	\begin{tabularx}{\textwidth}{lXXX}
		\toprule
		& \textbf{Female (n=""" + str(n_female) + r""")} & \textbf{Male (n=""" + str(
        n_male) + r""")} & \textbf{Total (n=""" + str(n_total) + r""")} \\
		\midrule

		Age (years)
		& """ + row_age[0] + r"""
		& """ + row_age[1] + r"""
		& """ + row_age[2] + r""" \\

		BMI (kg/m$^2$)
		& """ + row_bmi[0] + r"""
		& """ + row_bmi[1] + r"""
		& """ + row_bmi[2] + r""" \\

		\bottomrule
	\end{tabularx}
\end{table}
"""
    print(latex_table)


if __name__ == "__main__":
    filepath = r"C:\Users\Sofia\Desktop\Tese\dados\dxa_tanita.xlsx"
    out_plots = r"C:\Users\Sofia\Desktop\Tese\dados\blandaltman"
    df = pd.read_excel(filepath, index_col=0)

    # ICC analysis
    results_icc = compute_icc_pairs(df, VARIABLE_MAPPING, icc_type="ICC2")
    print(results_icc.to_string(index=False))

    # Generate characterization table of participants of ICC analysis
    generate_sample_characterization_table(df, age_col="Idade", gender_col="Gender", bmi_col="Body mass index (BMI)")

    # Bland Altman graphs
    plot_bland_altman_pairs(df, VARIABLE_MAPPING, show_plot=False, save_dir=out_plots)