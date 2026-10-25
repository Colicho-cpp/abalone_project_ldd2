
import pandas as pd
import optuna

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import root_mean_squared_error

# ==================================================
# 1. DATOS
# ==================================================

df = pd.read_csv("data/processed/abalone_clean.csv")

# Evitar data leakage
X = df.drop(columns=["age", "rings"])
y = df["age"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Estandarización solo con entrenamiento
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)


# ==================================================
# 2. FUNCIÓN OBJETIVO
# ==================================================

def objective(trial):

    capas = trial.suggest_categorical(
        "capas",
        [
            "32",
            "64",
            "128",
            "64-32",
            "128-64",
            "128-64-32",
            "256-128-64"
        ]
    )

    arquitectura = tuple(map(int, capas.split("-")))

    learning_rate = trial.suggest_float(
        "learning_rate",
        1e-5,
        1e-2,
        log=True
    )

    alpha = trial.suggest_float(
        "alpha",
        1e-6,
        1e-1,
        log=True
    )

    activacion = trial.suggest_categorical(
        "activation",
        ["relu", "tanh"]
    )

    batch_size = trial.suggest_categorical(
        "batch_size",
        [32, 64, 128, 256]
    )

    modelo = MLPRegressor(
        hidden_layer_sizes=arquitectura,
        activation=activacion,
        solver="adam",
        learning_rate_init=learning_rate,
        alpha=alpha,
        batch_size=batch_size,
        random_state=42,
        early_stopping=False
    )

    mejor_rmse = float("inf")

    # Entrenamiento incremental
    for epoca in range(200):

        modelo.partial_fit(X_train, y_train)

        pred = modelo.predict(X_val)
        rmse = root_mean_squared_error(y_val, pred)

        mejor_rmse = min(mejor_rmse, rmse)

        # Informar progreso a Optuna
        trial.report(rmse, step=epoca)

        # Poda de experimentos poco prometedores
        if trial.should_prune():
            raise optuna.TrialPruned()

    return rmse


# ==================================================
# 3. OPTIMIZACIÓN
# ==================================================

study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=25,
        interval_steps=5
    )
)

study.optimize(
    objective,
    n_trials=80,
    show_progress_bar=True
)


# ==================================================
# 4. RESULTADOS
# ==================================================

print("\n========== MEJOR CONFIGURACIÓN ==========")

for parametro, valor in study.best_params.items():
    print(f"{parametro}: {valor}")

print(f"\nMejor RMSE de validación: {study.best_value:.4f}")

print(f"Trials completados: {len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))}")
print(f"Trials podados: {len(study.get_trials(states=[optuna.trial.TrialState.PRUNED]))}")


# ==================================================
# 5. HISTORIAL DE OPTUNA
# ==================================================

study.trials_dataframe().to_csv(
    "results/optuna_trials_ema.csv",
    index=False
)

print("\nHistorial guardado en results/optuna_trials_ema.csv")
