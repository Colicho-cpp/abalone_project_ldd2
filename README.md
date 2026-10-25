# Abalone Project

Proyecto de Laboratorio de Datos II (Unidad 5). A partir de mediciones físicas de
abalones (dataset [Abalone, UCI](https://archive.ics.uci.edu/dataset/1/abalone)),
se construye un modelo de regresión para estimar la edad del molusco
(`age = rings + 1.5`). El foco está en que el flujo de trabajo sea **reproducible**:
entorno con versiones fijas, control de versiones con Git, ramas y Pull Requests,
y registro de datos, modelos y experimentos.

## Requisitos

- **Python 3.13.7**
- Git

## Reproducir el entorno

```bash
git clone <URL-del-repositorio>
cd abalone_project
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

Instalar las dependencias (versiones fijas):

```bash
pip install -r requirements.txt
```

Para usar el entorno en Jupyter / VS Code, seleccionar como kernel el intérprete
`.venv` del proyecto (`ipykernel` ya está incluido en `requirements.txt`).

## Estructura del proyecto

```
abalone_project/
├── data/
│   ├── raw/                  # datos originales (sin modificar)
│   ├── processed/            # datos limpios o transformados
│   └── data_log.json         # registro de transformaciones
├── src/
│   ├── preprocess.py         # transformaciones y limpieza
│   └── train_model.py        # entrenamiento y evaluación
├── results/
│   ├── metrics_v1.json       # métricas, hiperparámetros y semilla
│   └── experiments_log.csv   # registro de experimentos
├── models/                   # modelos serializados (.pkl)
├── .gitignore
├── requirements.txt
└── README.md
```

> Los archivos `.gitkeep` sólo sirven para que Git versione las carpetas vacías.

## Preprocesamiento

El script `src/preprocess.py` realiza las siguientes operaciones:

1. Carga el dataset Abalone original.
2. Crea la variable `age = rings + 1.5`.
3. Codifica `sex` mediante One-Hot Encoding.
4. Analiza valores atípicos mediante boxplots.
5. Elimina las observaciones con `height = 0`.
6. Genera `data/processed/abalone_clean.csv`.

Para ejecutarlo desde la raíz del proyecto:

    python src/preprocess.py

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

Para entrenar el modelo de regresión con el Pipeline (`StandardScaler` + `RandomForestRegressor`), evaluar su desempeño con los datos de test, generar el archivo serializado (`models/model_v1.pkl`) y guardar las métricas en formato JSON (`results/metrics_v1.json`), ejecutar:

python src/train_model.py

### Experimento v3 — KNN (rama `exp/knn`)

Entrena un Pipeline (`StandardScaler` + `KNeighborsRegressor`). Los hiperparámetros (`n_neighbors`, `weights`, `p`) se eligen con `GridSearchCV` (KFold de 5 folds, `random_state=42`) usando sólo los datos de entrenamiento, y luego se evalúa sobre test. Genera `models/model_v3.pkl`, `results/metrics_v3.json` y agrega una fila a `results/experiments_log.csv`:

    python src/train_knn.py

## Resultados

**Modelo v2 (SVR - kernel='rbf', C=10.0, epsilon=0.2):** MAE = 1.51 | RMSE = 2.21 | R² = 0.57

**Modelo v3 (KNN - n_neighbors=17, weights='distance', p=1):** MAE = 1.57 | RMSE = 2.33 | R² = 0.52

**Modelo MLP (MLPRegressor - (128, 64), Optuna):** MAE = 1.51 | RMSE = 2.15 | R² = 0.60

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
