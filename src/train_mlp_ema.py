
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


# ==================================================
# 1. CARGAR Y DIVIDIR LOS DATOS
# ==================================================

df = pd.read_csv("data/processed/abalone_clean.csv")

# Excluimos rings para evitar data leakage
X = df.drop(columns=["age", "rings"])
y = df["age"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==================================================
# 2. CREAR EL MLP CON LOS PARÁMETROS DE OPTUNA
# ==================================================

mlp = MLPRegressor(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    learning_rate_init=0.00235180773614519,
    alpha=0.0736284816059005,
    batch_size=32,
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=20,
    random_state=42
)

modelo = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", mlp)
])


# ==================================================
# 3. ENTRENAMIENTO Y EVALUACIÓN
# ==================================================

modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Entrenamiento completado en {mlp.n_iter_} épocas.")


# ==================================================
# 4. GUARDAR MODELO Y MÉTRICAS INDIVIDUALES
# ==================================================

Path("models").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

joblib.dump(
    modelo,
    "models/model_mlp_ema.pkl",
    compress=3
)

metricas = {
    "experiment_id": "exp_mlp_ema",
    "model_version": "mlp_ema_v1",
    "model_type": "MLPRegressor",
    "random_state_split": 42,
    "epochs_trained": mlp.n_iter_,
    "hyperparameters": mlp.get_params(),
    "metrics": {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }
}

with open("results/metrics_mlp_ema.json", "w", encoding="utf-8") as f:
    json.dump(metricas, f, indent=4)


# ==================================================
# 5. ACTUALIZAR EL HISTORIAL COMPARTIDO
# ==================================================

ruta = "results/experiments_log.csv"

if Path(ruta).exists():
    log = pd.read_csv(ruta)
else:
    log = pd.DataFrame()

nuevo = {
    "experiment_id": "exp_mlp_ema",
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model_version": "mlp_ema_v1",
    "model_type": "MLPRegressor",
    "hyperparameters": json.dumps(mlp.get_params()),
    "random_state": 42,
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2
}

# Evitar duplicados de nuestro experimento
if "experiment_id" in log.columns:
    log = log[log["experiment_id"] != "exp_mlp_ema"]

log = pd.concat([log, pd.DataFrame([nuevo])], ignore_index=True)
log.to_csv(ruta, index=False)


# ==================================================
# 6. GENERAR IMAGEN DE LA ARQUITECTURA
# ==================================================

capas = [X.shape[1], 128, 64, 1]
fig, ax = plt.subplots(figsize=(11, 6))

# Representamos hasta 10 neuronas por capa
for i, cantidad in enumerate(capas):

    visibles = min(cantidad, 10)
    posiciones = np.linspace(-2.5, 2.5, visibles)

    # Dibujar conexiones
    if i > 0:
        for y_actual in posiciones:
            for y_anterior in anteriores:
                ax.plot(
                    [i - 1, i],
                    [y_anterior, y_actual],
                    color="gray",
                    alpha=0.12,
                    linewidth=0.5
                )

    # Dibujar neuronas
    ax.scatter(
        [i] * visibles,
        posiciones,
        s=150,
        zorder=3
    )

    ax.text(
        i, -3.3,
        f"{cantidad} neuronas",
        ha="center"
    )

    anteriores = posiciones

ax.set_xlim(-0.5, 3.5)
ax.set_ylim(-4, 3.5)
ax.axis("off")
ax.set_title("Arquitectura MLP - Ema (10 → 128 → 64 → 1)")

plt.savefig(
    "results/mlp_ema.png",
    dpi=180,
    bbox_inches="tight"
)
plt.close()


print("Modelo, métricas, historial e imagen guardados correctamente.")
