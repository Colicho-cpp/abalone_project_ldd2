
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

# Cargar dataset
df = pd.read_csv("data/processed/abalone_clean.csv")

# Evitar data leakage
X = df.drop(columns=["age", "rings"])
y = df["age"]

# División 80/20, igual que el modelo original
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Mejor configuración encontrada por Optuna
modelo = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(
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
    ))
])

# Entrenar
modelo.fit(X_train, y_train)

print("Entrenamiento terminado.")
print(f"Épocas utilizadas: {modelo.named_steps['mlp'].n_iter_}")
