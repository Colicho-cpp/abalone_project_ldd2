"""Experimento v3: K-Nearest Neighbors (KNN) para predecir la edad de los abalones.

Ejecutar desde la raíz del proyecto:
    python src/train_knn.py

Genera:
    models/model_v3.pkl          pipeline entrenado (StandardScaler + KNeighborsRegressor)
    results/metrics_v3.json      métricas, hiperparámetros y semillas
    results/experiments_log.csv  agrega una fila con esta ejecución
"""
import csv
import json
import os
from datetime import datetime

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
MODEL_VERSION = "v3"
EXPERIMENT_ID = "exp_03"
MODEL_PATH = "models/model_v3.pkl"
METRICS_PATH = "results/metrics_v3.json"
LOG_PATH = "results/experiments_log.csv"
LOG_HEADER = ["experiment_id", "timestamp", "model_version", "n_estimators", "random_state",
              "MAE", "RMSE", "R2", "model_type", "hyperparameters"]

#Cargamos el dataset procesado
df = pd.read_csv("data/processed/abalone_clean.csv")

#'age' es el objetivo; se excluye 'rings' porque age = rings + 1.5 (sería data leakage)
X = df.drop(columns=["age", "rings"])
y = df["age"]

#Misma división que los demás experimentos para que las métricas sean comparables
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
print(f"Datos de entrenamiento: {X_train.shape}")
print(f"Datos de test: {X_test.shape}")

#KNN usa distancias, así que el escalado es imprescindible.
#Al estar dentro del Pipeline, el scaler se ajusta sólo con los datos de cada fold de train.
pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("model", KNeighborsRegressor()),
    ]
)

#Búsqueda de hiperparámetros con validación cruzada SOLO sobre train (el test no se toca)
param_grid = {
    "model__n_neighbors": list(range(3, 32, 2)),
    "model__weights": ["uniform", "distance"],
    "model__p": [1, 2],  # 1 = Manhattan, 2 = Euclídea
}
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
search = GridSearchCV(pipeline, param_grid, cv=cv, scoring="neg_root_mean_squared_error")
search.fit(X_train, y_train)

best_pipeline = search.best_estimator_
best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
print(f"Mejores hiperparámetros (CV): {best_params} | RMSE CV = {-search.best_score_:.4f}")

#Evaluación final sobre test
y_pred = best_pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Test -> MAE = {mae:.4f} | RMSE = {rmse:.4f} | R2 = {r2:.4f}")

#Guardamos el modelo comprimido
os.makedirs("models", exist_ok=True)
joblib.dump(best_pipeline, MODEL_PATH, compress=3)

#Guardamos métricas, hiperparámetros y semillas
os.makedirs("results", exist_ok=True)
metrics_data = {
    "model_version": MODEL_VERSION,
    "model_type": "KNeighborsRegressor",
    "target": "age",
    "hyperparameters": best_params,
    "hyperparameter_search": {
        "method": "GridSearchCV",
        "param_grid": {k.replace("model__", ""): v for k, v in param_grid.items()},
        "cv": "KFold(n_splits=5, shuffle=True)",
        "scoring": "neg_root_mean_squared_error",
        "best_cv_RMSE": -search.best_score_,
    },
    "metrics": {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    },
    "random_state_split": RANDOM_STATE,
    "random_state_cv": RANDOM_STATE,
}
with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics_data, f, indent=4)

#Registramos la ejecución en el log de experimentos.
#Si el encabezado todavía no tiene las columnas genéricas model_type/hyperparameters, se agregan.
if os.path.exists(LOG_PATH):
    with open(LOG_PATH, encoding="utf-8") as f:
        lines = f.read().splitlines()
    if lines and lines[0].split(",") != LOG_HEADER:
        lines[0] = ",".join(LOG_HEADER)
        with open(LOG_PATH, "w", encoding="utf-8", newline="") as f:
            f.write("\n".join(lines) + "\n")
else:
    with open(LOG_PATH, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(LOG_HEADER) + "\n")

row = [
    EXPERIMENT_ID,
    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    MODEL_VERSION,
    "",  # n_estimators no aplica a KNN
    RANDOM_STATE,
    mae,
    rmse,
    r2,
    "KNeighborsRegressor",
    json.dumps(best_params),
]
with open(LOG_PATH, "a", encoding="utf-8", newline="") as f:
    csv.writer(f, lineterminator="\n").writerow(row)

print("Experimento registrado en", LOG_PATH)
