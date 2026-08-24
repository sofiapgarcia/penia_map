from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURAÇÕES - CAMINHOS
# ============================================================

# Excel de referência usado para ajustar o polinómio (Age, M, F)
EXCEL_PATH = Path(r"C:\Users\Sofia\Desktop\Tese\dados\bone_mass.xlsx")

# Pasta onde são guardados os resultados do ajuste (métricas, figura, coeficientes)
OUTPUT_DIR = Path(r"C:\Users\Sofia\Desktop\Tese\dados")

# Excel novo onde se quer estimar a idade óssea (deve ter Age, Gender, Estimated bone mass kg)
NEW_EXCEL_PATH = Path(r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\final_dataframe.xlsx")

# Nome do ficheiro de saída, guardado na MESMA pasta do NEW_EXCEL_PATH
NEW_EXCEL_OUTPUT_PATH = NEW_EXCEL_PATH.with_name(
    NEW_EXCEL_PATH.stem + "_com_idade_ossea.xlsx"
)

# Nomes das colunas no Excel novo (ajustar se forem diferentes)
COL_GENDER = "Gender"
COL_BONE_MASS_KG = "Estimated bone mass kg"
COL_CHRON_AGE = "Age (target)"
COL_NEW_AGE = "age_bone_mass"

# Intervalo de idades para o qual as equações são válidas.
# Soluções (raízes) fora deste intervalo são descartadas.
AGE_MIN_VALID = 20
AGE_MAX_VALID = 85

DEGREES_TO_TEST = range(1, 6)   # graus a testar
FINAL_DEGREE = 4                # grau usado no modelo final

SEXES = {"M": "Male", "F": "Female"}
GENDER_CODE_TO_SEX = {1: "M", 2: "F"}


# ============================================================
# FUNÇÕES - AJUSTE DO POLINÓMIO
# ============================================================

def load_data(excel_path):
    """Lê as colunas Age, M e F e converte para numérico."""
    df = pd.read_excel(excel_path, usecols=[0, 1, 2])
    df.columns = ["Age", "M", "F"]
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.sort_values("Age")


def get_age_mass(df, sex_column):
    """Devolve (age, bone_mass) sem NaNs para um dado sexo."""
    sub = df[["Age", sex_column]].dropna()
    return sub["Age"].to_numpy(), sub[sex_column].to_numpy()


def fit_polynomial(x, y, degree):
    """Ajusta um polinómio e devolve coeficientes, modelo, R² e RMSE."""
    coeffs = np.polyfit(x, y, degree)
    poly = np.poly1d(coeffs)
    residuals = y - poly(x)
    r2 = 1 - np.sum(residuals ** 2) / np.sum((y - y.mean()) ** 2)
    rmse = np.sqrt(np.mean(residuals ** 2))
    return coeffs, poly, r2, rmse


def evaluate_degrees(age, mass, sex_name):
    """Testa vários graus e devolve lista de resultados (dicts)."""
    rows = []
    print(f"\n{sex_name}\n{'-'*60}")
    print(f"Observações: {len(age)}")

    for degree in DEGREES_TO_TEST:
        if degree >= len(age):
            print(f"Grau {degree}: ignorado (poucas observações).")
            continue
        _, _, r2, rmse = fit_polynomial(age, mass, degree)
        rows.append({"Sex": sex_name, "Degree": degree, "R2": r2, "RMSE": rmse})
        print(f"Grau {degree}: R² = {r2:.5f}, RMSE = {rmse:.5f}")

    return rows


def fit_final_model(age, mass, sex_name, degree):
    """Ajusta o modelo final para um sexo e devolve um dicionário com tudo."""
    if degree >= len(age):
        raise ValueError(f"FINAL_DEGREE demasiado alto para {sex_name}.")
    coeffs, poly, r2, rmse = fit_polynomial(age, mass, degree)
    return {
        "name": sex_name, "age": age, "mass": mass,
        "coefficients": coeffs, "polynomial": poly, "r2": r2, "rmse": rmse,
    }


def print_final_model(model, degree):
    print(f"\n{model['name']}\n{'-'*60}")
    print(f"Degree: {degree}")
    print(f"R²:     {model['r2']:.6f}")
    print(f"RMSE:   {model['rmse']:.6f}")
    print("\nPolinómio:")
    print(model["polynomial"])


def coefficients_to_df(models, degree):
    rows = [
        {"Sex": m["name"], "Power": power, "Coefficient": coef}
        for m in models.values()
        for power, coef in zip(range(degree, -1, -1), m["coefficients"])
    ]
    return pd.DataFrame(rows)


def plot_models(models, degree, output_path):
    plt.figure(figsize=(10, 6))

    for m in models.values():
        age_curve = np.linspace(m["age"].min(), m["age"].max(), 500)
        plt.scatter(m["age"], m["mass"], label=f"{m['name']} observado", zorder=3)
        plt.plot(age_curve, m["polynomial"](age_curve), linewidth=2,
                  label=f"{m['name']} polinómio (grau {degree})")

    plt.xlabel("Idade (anos)")
    plt.ylabel("Massa óssea")
    plt.title("Ajuste polinomial da massa óssea em função da idade")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


# ============================================================
# FUNÇÕES - ESTIMATIVA DA IDADE ÓSSEA (NOVOS DADOS)
# ============================================================

def estimate_bone_age(target_mass, polynomial, chronological_age):
    """
    Resolve polynomial(age) = target_mass e devolve a solução real
    mais próxima de chronological_age.

    Se não houver raízes reais, devolve np.nan.
    """
    if pd.isna(target_mass) or pd.isna(chronological_age):
        return np.nan

    # polynomial(age) - target_mass = 0
    shifted_coeffs = polynomial.coefficients.copy()
    shifted_coeffs[-1] -= target_mass

    roots = np.roots(shifted_coeffs)

    # manter apenas raízes (praticamente) reais
    real_roots = roots[np.isclose(roots.imag, 0, atol=1e-6)].real

    # manter apenas raízes dentro do intervalo de idades válido
    # (as equações só são válidas entre AGE_MIN_VALID e AGE_MAX_VALID)
    valid_roots = real_roots[
        (real_roots >= AGE_MIN_VALID) & (real_roots <= AGE_MAX_VALID)
    ]

    if len(valid_roots) == 0:
        return np.nan

    # escolher, de entre as raízes válidas, a mais próxima da idade cronológica
    closest_root = valid_roots[
        np.argmin(np.abs(valid_roots - chronological_age))
    ]

    return closest_root


def add_age_bone_mass_column(df, final_models):
    """
    Adiciona a coluna age_bone_mass ao DataFrame, calculada a partir de
    COL_BONE_MASS_KG (convertido para gramas) e do polinómio ajustado
    correspondente ao Gender de cada linha.
    """
    df = df.copy()
    estimated_ages = []

    for _, row in df.iterrows():

        gender_code = row.get(COL_GENDER)
        mass_kg = row.get(COL_BONE_MASS_KG)
        chron_age = row.get(COL_CHRON_AGE)

        sex_column = GENDER_CODE_TO_SEX.get(gender_code)

        if sex_column is None or sex_column not in final_models:
            estimated_ages.append(np.nan)
            continue

        mass_g = mass_kg * 1000  # kg -> gramas

        polynomial = final_models[sex_column]["polynomial"]

        age_estimate = estimate_bone_age(
            target_mass=mass_g,
            polynomial=polynomial,
            chronological_age=chron_age
        )

        estimated_ages.append(age_estimate)

    df[COL_NEW_AGE] = estimated_ages

    return df


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # PARTE 1: ajuste do polinómio com os dados de referência
    # --------------------------------------------------------

    df = load_data(EXCEL_PATH)
    print(f"\nLinhas no Excel: {len(df)}")
    print(f"Intervalo de idades: {df['Age'].min():.2f} - {df['Age'].max():.2f} anos")

    print("\nResultados do ajuste polinomial:\n" + "=" * 60)
    all_results = []
    data_by_sex = {}

    for sex_col, sex_name in SEXES.items():
        age, mass = get_age_mass(df, sex_col)
        data_by_sex[sex_col] = (age, mass)
        all_results += evaluate_degrees(age, mass, sex_name)

    pd.DataFrame(all_results).to_excel(
        OUTPUT_DIR / "polynomial_fit_metrics.xlsx", index=False
    )

    print("\n\nMODELOS POLINOMIAIS FINAIS\n" + "=" * 60)
    final_models = {}
    for sex_col, sex_name in SEXES.items():
        age, mass = data_by_sex[sex_col]
        model = fit_final_model(age, mass, sex_name, FINAL_DEGREE)
        final_models[sex_col] = model
        print_final_model(model, FINAL_DEGREE)

    coefficients_to_df(final_models, FINAL_DEGREE).to_excel(
        OUTPUT_DIR / "final_polynomial_coefficients.xlsx", index=False
    )

    figure_path = OUTPUT_DIR / "polynomial_fit_M_F.png"
    plot_models(final_models, FINAL_DEGREE, figure_path)

    print(f"\nFigura guardada em:\n{figure_path}")
    print(f"\nResultados guardados em:\n{OUTPUT_DIR.resolve()}")

    # --------------------------------------------------------
    # PARTE 2: estimar idade óssea nos novos dados
    # --------------------------------------------------------

    print("\n\nESTIMATIVA DA IDADE ÓSSEA (novos dados)\n" + "=" * 60)

    new_df = pd.read_excel(NEW_EXCEL_PATH)

    new_df = add_age_bone_mass_column(new_df, final_models)

    new_df.to_excel(NEW_EXCEL_OUTPUT_PATH, index=False)

    n_estimated = new_df[COL_NEW_AGE].notna().sum()

    print(f"Linhas processadas: {len(new_df)}")
    print(f"Idades ósseas estimadas com sucesso: {n_estimated}")
    print(f"\nFicheiro guardado em:\n{NEW_EXCEL_OUTPUT_PATH}")
    # --------------------------------------------------------
    # Diagnóstico: sobrepor os dados novos à curva de referência
    # --------------------------------------------------------

    diag_df = pd.read_excel(NEW_EXCEL_PATH)

    plt.figure(figsize=(10, 6))

    # Curvas de referência (M e F)
    for sex_col, model in final_models.items():
        age_curve = np.linspace(AGE_MIN_VALID, AGE_MAX_VALID, 500)
        plt.plot(age_curve, model["polynomial"](age_curve),
                 linewidth=2, label=f"Referência {model['name']}")

    # Pontos novos, coloridos por sexo
    for gender_code, sex_col in GENDER_CODE_TO_SEX.items():
        subset = diag_df[diag_df[COL_GENDER] == gender_code]
        mass_g = subset[COL_BONE_MASS_KG] * 1000
        plt.scatter(subset[COL_CHRON_AGE], mass_g, alpha=0.5,
                    label=f"Amostra {SEXES[sex_col]}", zorder=3)

    plt.xlabel("Idade (anos)")
    plt.ylabel("Massa óssea (g)")
    plt.title("Dados novos vs. curva de referência")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "diagnostico_dados_vs_referencia.png", dpi=300, bbox_inches="tight")
    plt.show()

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
