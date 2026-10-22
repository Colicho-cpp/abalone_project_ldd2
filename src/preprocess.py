import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(
    "data/raw/abalone_raw.csv",
    header=None,
    names=["sex", "length", "diameter", "height",
           "whole_weight", "shucked_weight",
           "viscera_weight", "shell_weight", "rings"]
)
print(f"Dataset original: {df.shape}")


# Crear la variable objetivo: edad estimada -------------
df["age"] = df["rings"] + 1.5

# Codificar sex mediante One-Hot Encoding ---------------
df = pd.get_dummies(
    df,
    columns=["sex"],
    dtype=int
)

# Buscamos outliers y graficamos boxplots ----------------

variables = ["length", "diameter", "height","whole_weight", "shucked_weight","viscera_weight", "shell_weight", "rings"]
fig, axes = plt.subplots(2, 4, figsize=(18, 9))

for ax, columna in zip(axes.flat, variables):
    sns.boxplot(y=df[columna],ax=ax,color="lightblue",flierprops={"marker": "o","markerfacecolor": "red","markersize": 4,"alpha": 0.6})
    ax.set_title(columna)
    ax.set_ylabel("")
    ax.grid(axis="y", alpha=0.3)

plt.suptitle("Análisis de Outliers - Dataset Abalone", fontsize=17)
plt.tight_layout()
plt.show()

variables = [
    "length", "diameter", "height",
    "whole_weight", "shucked_weight",
    "viscera_weight", "shell_weight", "rings"
]

# Crear y mostrar tablas ---------------------------------------
mayores = pd.DataFrame({
    columna: df[columna].nlargest(5).values
    for columna in variables
})

menores = pd.DataFrame({
    columna: df[columna].nsmallest(5).values
    for columna in variables
})

mayores.index += 1
menores.index += 1

print("\n========================= 5 VALORES MÁS ALTOS =========================")
print(mayores.to_string())

print("\n========================= 5 VALORES MÁS BAJOS =========================")
print(menores.to_string())

# Consideramos todos los valores outliers como legitimos excepto los de "Height" con valor 0
# Evitamos riesgos y simplemente borramos las filas ya que son solo 2

filas_antes = len(df)

df = df[df["height"] != 0].copy()

print(f"Filas eliminadas: {filas_antes - len(df)}")
print(f"Filas restantes: {len(df)}")

# Guardar dataset procesado -------------------------------
df.to_csv(
    "data/processed/abalone_clean.csv",
    index=False
)

print(f"Dataset procesado guardado: {df.shape}")