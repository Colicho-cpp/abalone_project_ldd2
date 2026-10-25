import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import joblib
from datetime import datetime
import os
import json

#Cargamos datos limpios
df = pd.read_csv("data/processed/abalone_clean.csv")

print("Columnas disponibles en el CSV:")
print(df.columns.tolist())

#Separamos features y target
X = df.drop(columns=["rings", "age"])
y = df["rings"]

#Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Pipeline con Gradient Boosting
pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('model', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
])

#Entrenamos el modelo
print("Entrenando el modelo v4 con Gradient Boosting...")
pipeline.fit(X_train, y_train)

#Evaluamos el modelo
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Métricas - MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")

#Guardar modelo v4
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/model_v4.pkl", compress=3)

#Guardar métricas v4
os.makedirs("results", exist_ok=True)
experiment_results = {
    "model_version": "v4",
    "hyperparameters": {
        "n_estimators": 100,          
        "learning_rate": 0.1,
        "random_state": 42
    },
    "metrics": {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    },
    "random_state_split": 42
}

with open("results/metrics_v4.json", "w") as f:
    json.dump(experiment_results, f, indent=4)

#Registrar la ejecución en el log de experimentos
log_file = "results/experiments_log.csv"
#Mismas columnas y orden que el encabezado de experiments_log.csv
log_data = {
    "experiment_id": ["exp_04"],
    "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    "model_version": ["v4"],
    "n_estimators": [100],
    "random_state": [42],
    "MAE": [mae],
    "RMSE": [rmse],
    "R2": [r2],
    "model_type": ["GradientBoostingRegressor"],
    "hyperparameters": [json.dumps(experiment_results["hyperparameters"])],
}

log_df = pd.DataFrame(log_data)
log_df.to_csv(log_file, mode="a", header=not os.path.exists(log_file), index=False)

print("¡Modelo v4 con Gradient Boosting entrenado y registrado con éxito!")