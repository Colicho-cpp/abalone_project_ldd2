# Abalone Project

Proyecto de Laboratorio de Datos II (Unidad 5). A partir de mediciones físicas de
abalones (dataset [Abalone, UCI](https://archive.ics.uci.edu/dataset/1/abalone)),
se construye un modelo de regresión para estimar la edad del molusco
(`age = rings + 1.5`). El foco está en que el flujo de trabajo sea **reproducible**:
entorno con versiones fijas, control de versiones con Git, ramas y Pull Requests,
y registro de datos, modelos y experimentos.

## Requisitos

- **Python 3.13.7** (se recomienda usar la misma versión 3.13)
- Git (o GitHub Desktop)

## Reproducir el entorno

Clonar el repositorio (la carpeta que se crea se llama `abalone_project_ldd2`):

```bash
git clone https://github.com/Colicho-cpp/abalone_project_ldd2.git
cd abalone_project_ldd2
python -m venv .venv
```

Activar el entorno:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Windows (CMD)
.venv\Scripts\activate.bat
# Linux / macOS
source .venv/bin/activate
```

> Si PowerShell bloquea `Activate.ps1` por la política de ejecución, usar la terminal CMD
> (`.venv\Scripts\activate.bat`) o la terminal integrada de VS Code.

Instalar las dependencias (versiones fijas):

```bash
pip install -r requirements.txt
```

Para usar el entorno en Jupyter / VS Code, seleccionar como kernel el intérprete
`.venv` del proyecto (`ipykernel` ya está incluido en `requirements.txt`).

## Estructura del proyecto

```
abalone_project_ldd2/
├── data/
│   ├── raw/
│   │   └── abalone_raw.csv          # datos originales (sin modificar)
│   ├── processed/
│   │   └── abalone_clean.csv        # datos limpios (generado por preprocess.py)
│   └── data_log.json                # registro de transformaciones
├── src/
│   ├── preprocess.py                # limpieza y transformaciones
│   ├── train_model.py               # experimento v2: SVR
│   ├── train_knn.py                 # experimento v3: KNN
│   ├── train_model_gradb.py         # experimento v4: Gradient Boosting
│   ├── train_mlp_ema.py             # experimento MLP (MLPRegressor)
│   └── optuna_mlp_ema.py            # búsqueda de hiperparámetros del MLP (opcional)
├── results/
│   ├── metrics_v1.json ... metrics_v4.json, metrics_mlp_ema.json
│   ├── optuna_trials_ema.csv        # historial de la búsqueda con Optuna
│   └── experiments_log.csv          # registro de todos los experimentos
├── models/                          # modelos serializados: model_v1..v4.pkl, model_mlp_ema.pkl
├── .gitignore
├── requirements.txt
└── README.md
```

> Los archivos `.gitkeep` sólo sirven para que Git versione las carpetas vacías.

## Cómo ejecutar el proyecto

**Todos los scripts se ejecutan desde la raíz del proyecto** (la carpeta `abalone_project_ldd2`),
con el entorno activado, y en este orden:

1. Preprocesamiento: `python src/preprocess.py`
2. Entrenamientos (son independientes entre sí, en cualquier orden):
   `python src/train_model.py`, `python src/train_knn.py`,
   `python src/train_model_gradb.py`, `python src/train_mlp_ema.py`

Todos los experimentos usan la misma división train/test (`test_size=0.2`, `random_state=42`)
y excluyen `rings` de las variables de entrada para evitar *data leakage*.

## Preprocesamiento

El script `src/preprocess.py` realiza las siguientes operaciones:

1. Carga el dataset Abalone original (`data/raw/abalone_raw.csv`).
2. Crea la variable `age = rings + 1.5`.
3. Codifica `sex` mediante One-Hot Encoding.
4. Analiza valores atípicos mediante boxplots.
5. Elimina las observaciones con `height = 0`.
6. Genera `data/processed/abalone_clean.csv`.

Para ejecutarlo desde la raíz del proyecto:

    python src/preprocess.py

> **Importante:** el script abre una ventana con los boxplots de outliers y **no continúa
> hasta que se cierra esa ventana**. Al cerrarla, termina de generar el CSV procesado.

Las transformaciones están documentadas en `data/data_log.json`.

## Ejercicio 2 — Preguntas teóricas

### 1. ¿Qué información de data_log.json es imprescindible para reconstruir el dataset procesado? ¿Nuestro registro alcanza?
Sí, el registro permite entender cómo se procesaron los datos. Para reproducirlos exactamente, también necesitamos el archivo original, el código preprocess.py y las dependencias correspondientes

### 2. ¿Qué pasaría si el dataset pesara varios GB o cambiara todas las semanas?
Guardar los datos directamente en Git sería problemático porque:
- El repositorio crecería demasiado.
- Cada nueva versión ocuparía espacio.
- Clonar y descargar el proyecto sería cada vez más lento.

Una alternativa sería separar el código de los datos, utilizando .gitignore para evitar almacenar los datasets directamente en Git y servicios externos como Amazon S3 para guardarlos.

Además, podríamos utilizar herramientas como DVC para gestionar las versiones mediante referencias y hashes, manteniendo la trazabilidad sin almacenar los archivos completos en Git.

## Entrenamiento y evaluación

Cada script entrena un Pipeline (`StandardScaler` + modelo), lo evalúa sobre test, guarda el
modelo comprimido en `models/`, sus métricas en `results/` y **agrega una fila** a
`results/experiments_log.csv`.

### Modelo v1 — Random Forest (Ejercicio 3)

El modelo v1 (`StandardScaler` + `RandomForestRegressor`, `n_estimators=100`, `random_state=42`)
se entrenó originalmente con `src/train_model.py`. Ese script fue reutilizado luego para el
experimento v2, por lo que **el código actual ya no regenera** `models/model_v1.pkl` ni
`results/metrics_v1.json`; se conservan en el repositorio como resultado histórico.

### Experimento v2 — SVR (rama `exp/algoritmo-gero`)

Entrena un Pipeline (`StandardScaler` + `SVR` con `kernel='rbf'`, `C=10.0`, `epsilon=0.2`).
Genera `models/model_v2.pkl` y `results/metrics_v2.json`:

    python src/train_model.py

### Experimento v3 — KNN (rama `exp/knn`)

Entrena un Pipeline (`StandardScaler` + `KNeighborsRegressor`). Los hiperparámetros (`n_neighbors`, `weights`, `p`) se eligen con `GridSearchCV` (KFold de 5 folds, `random_state=42`) usando sólo los datos de entrenamiento, y luego se evalúa sobre test. Genera `models/model_v3.pkl` y `results/metrics_v3.json`:

    python src/train_knn.py

### Experimento v4 — Gradient Boosting (rama `exp/hiperparametros-iara`)

Entrena un Pipeline (`StandardScaler` + `GradientBoostingRegressor` con `n_estimators=100`, `learning_rate=0.1`, `random_state=42`). Genera `models/model_v4.pkl` y `results/metrics_v4.json`:

    python src/train_model_gradb.py

### Experimento MLP — MLPRegressor (rama `exp/mlp-emasavat`)

Entrena un Pipeline (`StandardScaler` + `MLPRegressor`) con los hiperparámetros encontrados por
Optuna (ya fijados en el script). Genera `models/model_mlp_ema.pkl`,
`results/metrics_mlp_ema.json` y el gráfico `results/mlp_ema.png`:

    python src/train_mlp_ema.py

**Opcional:** para repetir la búsqueda de hiperparámetros con Optuna (80 trials, puede tardar
varios minutos) y regenerar `results/optuna_trials_ema.csv`:

    python src/optuna_mlp_ema.py


## Resultados

**Modelo v2 (SVR - kernel='rbf', C=10.0, epsilon=0.2):** MAE = 1.51 | RMSE = 2.21 | R² = 0.57

**Modelo v3 (KNN - n_neighbors=17, weights='distance', p=1):** MAE = 1.57 | RMSE = 2.33 | R² = 0.52

**Modelo MLP (MLPRegressor - (128, 64), Optuna):** MAE = 1.51 | RMSE = 2.15 | R² = 0.60

**Modelo v4 (Gradient Boosting - n_estimators=100, learning_rate=0.1):** MAE = 1.56 | RMSE = 2.24 | R² = 0.56

### MLPRegressor — Emanuel Savat

Se entrenó una red neuronal multicapa para predecir la edad de los abalones, excluyendo `rings` para evitar *data leakage*.

**Optimización con Optuna:**
- 80 trials, utilizando `MedianPruner`.
- Arquitectura seleccionada: (128, 64).
- Activación: ReLU.
- Optimizador: Adam.
- Learning rate: 0.002352.
- Regularización (alpha): 0.073628.
- Batch size: 32.
- Early Stopping habilitado.

**Resultados de evaluación:**

| Modelo | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest corregido | 1.5880 | 2.2476 | 0.5573 |
| MLPRegressor | **1.5081** | **2.1462** | **0.5963** |

El modelo entrenado se encuentra en `models/model_mlp_ema.pkl`, las métricas en `results/metrics_mlp_ema.json` y el historial de Optuna en `results/optuna_trials_ema.csv`.
