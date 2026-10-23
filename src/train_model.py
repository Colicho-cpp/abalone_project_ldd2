import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import joblib
import os
import json
from datetime import datetime

#cargamos el dataset procesado
df=pd.read_csv("data/processed/abalone_clean.csv")

#Separamos características (X) y objetivo (y)
#'age' es la variable que queremos predecir
X = df.drop(columns=["age", "rings"])
y = df["age"]

#Dividimos en entrenamiento y test fijando la semilla (random_state)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Datos de entrenamiento: {X_train.shape}")
print(f"Datos de test: {X_test.shape}")

#Creamos el Pipeline asegurando que el escalador solo aprenda de train
pipeline=Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(n_estimators=100, random_state=42)),
    ]
)

#Entrenamos el pipeline completo con los datos de entrenamiento
pipeline.fit(X_train, y_train)

#El modelo predice sobre los datos de test
y_pred = pipeline.predict(X_test)

#Se calculan las métricas
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/model_v1.pkl", compress=3)

#Guardamos las métricas, hiperparámetros y semilla en un JSON

os.makedirs("results", exist_ok=True)
metrics_data = {
    "model_version": "v1",
    "hyperparameters": {
        "n_estimators": 100,
        "random_state": 42
    },
    "metrics": {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    },
    "random_state_split": 42
}

with open("results/metrics_v1.json", "w") as f:
    json.dump(metrics_data, f, indent=4)

#registramos la ejecución en results/experiments_log.csv
log_file="results/experiments_log.csv"
log_data={
    "experiment_id": ["exp_01"],
    "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    "model_version": ["v1"],
    "n_estimators": [100],
    "random_state": [42],
    "MAE": [mae],
    "RMSE": [rmse],
    "R2": [r2]
}

log_df=pd.DataFrame(log_data)
if not os.path.exists(log_file):
    log_df.to_csv(log_file, index=False)
else:
    log_df.to_csv(log_file, mode="a", header=False, index=False)

print("¡Experimento registrado con éxito en el log!")