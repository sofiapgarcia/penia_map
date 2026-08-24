import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ignorar este codigo
excel_file = r"C:\Users\Sofia\Desktop\Tese\dados\compracao idade tanita e idade ossea.xlsx"

x_column = "Estimated metabolic age"

# Ler Excel
df = pd.read_excel(excel_file)

if x_column not in df.columns:
    raise ValueError(f"A coluna '{x_column}' não existe no ficheiro.")


# Converter X para numérico
x = pd.to_numeric(df[x_column], errors="coerce")

resultados = []

for coluna in df.columns:

    if coluna == x_column:
        continue

    # Ignorar colunas não numéricas
    y = pd.to_numeric(df[coluna], errors="coerce")

    dados = pd.DataFrame({
        "x": x,
        "target": y
    }).dropna()

    if len(dados) < 2:
        continue

    X = dados["x"].values.reshape(-1, 1)
    Y = dados["target"].values

    # Regressão Linear
    modelo = LinearRegression()
    modelo.fit(X, Y)

    y_pred = modelo.predict(X)

    r2 = r2_score(Y, y_pred)
    slope = modelo.coef_[0]
    intercept = modelo.intercept_

    resultados.append({
        "Variável": coluna,
        "R²": r2,
        "Inclinação": slope,
        "Interceção": intercept
    })

    # ---------- Gráfico ----------
    plt.figure(figsize=(8,6))

    plt.scatter(X, Y, s=40)

    ordem = np.argsort(X[:,0])

    plt.plot(
        X[:,0][ordem],
        y_pred[ordem],
        linewidth=2
    )

    plt.xlabel(x_column)
    plt.ylabel(coluna)
    plt.title(coluna)

    texto = (
        f"target = {slope:.4f}x + {intercept:.4f}\n"
        f"R² = {r2:.4f}"
    )

    plt.text(
        0.05,
        0.95,
        texto,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(facecolor="white")
    )

    plt.grid(True)
    plt.show()



# Resultados
resultados = pd.DataFrame(resultados)

resultados = resultados.sort_values(
    by="R²",
    ascending=False
)

print("\n==============================")
print("RESULTADOS")
print("==============================")
print(resultados)

print("\nMelhor ajuste:")
print(resultados.iloc[0])