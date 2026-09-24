import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Ler o dataframe ---------------------------------------------------
# Ajusta o caminho do ficheiro conforme necessário
caminho_ficheiro = r"C:\Users\Sofia\Desktop\Tese\dados\maio\certos\validos\final_dataframe_filtered_without_mean_variables.xlsx"  # <-- muda para o teu ficheiro
df = pd.read_excel(caminho_ficheiro)

# --- 2. Preparar os dados ---------------------------------------------------
coluna_idade = "Age (y)"
coluna_genero = "Gender"

mapa_genero = {1: "Male", 0: "Female"}
df["Gender_label"] = df[coluna_genero].map(mapa_genero)

# --- 3. Criar o histograma sobreposto -----------------------------------
# Os dois géneros são desenhados no mesmo gráfico, com transparência (alpha)
# para se conseguir ver a sobreposição.
fig, eixo = plt.subplots(figsize=(9, 6))

generos = ["Male", "Female"]
cores = ["#4C72B0", "#DD8452"]

# Bins comuns para os dois grupos, definidos em anos (para serem
# diretamente comparáveis). Muda "intervalo_anos" para o tamanho que quiseres
# (ex.: 1 = um bin por cada ano; 5 = bins de 5 em 5 anos).
intervalo_anos = 10

idade_min = df[coluna_idade].min()
idade_max = df[coluna_idade].max()

# Arredondar os limites para múltiplos do intervalo, para bins "bonitos"
idade_min_arred = (idade_min // intervalo_anos) * intervalo_anos
idade_max_arred = ((idade_max // intervalo_anos) + 1) * intervalo_anos

bins = range(int(idade_min_arred), int(idade_max_arred) + intervalo_anos, intervalo_anos)

for genero, cor in zip(generos, cores):
    dados_genero = df.loc[df["Gender_label"] == genero, coluna_idade].dropna()
    eixo.hist(
        dados_genero,
        bins=bins,
        color=cor,
        edgecolor="black",
        alpha=0.55,
        label=genero,
    )

eixo.set_title("Distribuição de idades por género", fontsize=14)
eixo.set_xlabel("Idade (anos)")
eixo.set_ylabel("Número de pessoas")
eixo.legend(title="Gender")
fig.tight_layout()

# --- 4. Guardar / mostrar o gráfico -----------------------------------------
fig.savefig("distribuicao_idade_genero.png", dpi=150)
plt.show()
