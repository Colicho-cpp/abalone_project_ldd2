
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

# 1. CARGA DEL DATASET

columnas = ["sex", "length", "diameter", "height","whole_weight", "shucked_weight","viscera_weight", "shell_weight", "rings"]
df = pd.read_csv("data/raw/abalone_raw.csv",header=None,names=columnas)
print(f"Dataset original: {df.shape}")


# 2. TRANSFORMACIÓN DE VARIABLES

# Crear la variable objetivo: edad estimada
df["age"] = df["rings"] + 1.5

# Codificar sex mediante One-Hot Encoding
df = pd.get_dummies(
    df,
    columns=["sex"],
    dtype=int
)


# 3. DETECCIÓN DE OUTLIERS: IQR

variables = ["length", "diameter", "height","whole_weight", "shucked_weight","viscera_weight", "shell_weight", "rings"]

print("\n--- ANÁLISIS DE OUTLIERS (IQR) ---")

for columna in variables:

    Q1 = df[columna].quantile(0.25)
    Q3 = df[columna].quantile(0.75)
    IQR = Q3 - Q1

    limite_inf = Q1 - 1.5 * IQR
    limite_sup = Q3 + 1.5 * IQR

    mascara = (
        (df[columna] < limite_inf) |
        (df[columna] > limite_sup)
    )

    print(
        f"{columna:18} | "
        f"Outliers: {mascara.sum():4} | "
        f"Límites: [{limite_inf:.3f}, {limite_sup:.3f}]"
    )


# 4. DETECCIÓN DE ANOMALÍAS: LOF

print("\n--- ANÁLISIS DE OUTLIERS (LOF) ---")

# Estandarización temporal para calcular distancias
X = StandardScaler().fit_transform(df[variables])

lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.01
)

predicciones = lof.fit_predict(X)

# Obtener observaciones anómalas
df_lof = df.loc[predicciones == -1].copy()

df_lof["lof_score"] = (
    -lof.negative_outlier_factor_[predicciones == -1]
)

df_lof = df_lof.sort_values(
    "lof_score",
    ascending=False
)

print(f"Anomalías detectadas: {len(df_lof)}")
print(df_lof.head(15).to_string())


# 5. TRATAMIENTO DE OUTLIERS: CLIPPING

print("\n--- TRATAMIENTO DE OUTLIERS ---")

# Conservar una copia de height para comparar
height_original = df["height"].copy()

# Limitar alturas a valores entre 0.010 y 0.250
df["height"] = df["height"].clip(
    lower=0.010,
    upper=0.250
)

# Contar cuántas observaciones fueron modificadas
modificados = (height_original != df["height"]).sum()

print(f"Valores de height modificados: {modificados}")
print(f"Filas conservadas: {len(df)}")


# 6. RESUMEN FINAL

print("\n--- RESUMEN DEL PREPROCESAMIENTO ---")
print(f"Dimensiones finales: {df.shape}")
print(f"Valores nulos: {df.isna().sum().sum()}")
print("\nPrimeras filas:")
print(df.head())
