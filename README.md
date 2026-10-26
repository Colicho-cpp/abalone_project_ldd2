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
> (`.venv\Scripts\activate.bat`), la terminal integrada de VS Code, o ejecutar previamente en PowerShell:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

Instalar las dependencias (versiones fijas):

```bash
pip install -r requirements.txt
```

Para usar el entorno en Jupyter / VS Code, seleccionar como kernel el intérprete
`.venv` del proyecto (`ipykernel` ya está incluido en `requirements.txt`).

## Estructura del proyecto

```text
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

```bash
python src/preprocess.py
```

> **Importante:** el script abre una ventana con los boxplots de outliers y **no continúa
> hasta que se cierra esa ventana**. Al cerrarla, termina de generar el CSV procesado.

Las transformaciones están documentadas en `data/data_log.json`.

---

## Ejercicio 2 — Preguntas teóricas

### 1. ¿Qué información de data_log.json es imprescindible para reconstruir el dataset procesado? ¿Nuestro registro alcanza?

Para que otra persona pueda reconstruir exactamente el dataset procesado partiendo del original, el archivo `data_log.json` debe especificar de manera inequívoca:
- La ruta o referencia exacta del archivo de origen (`source`: `data/raw/abalone_raw.csv`).
- El orden secuencial de las transformaciones aplicadas.
- Las fórmulas exactas de las variables derivadas (`age = rings + 1.5`).
- El método y esquema de codificación de variables categóricas (One-Hot Encoding sobre `sex` generando `sex_F`, `sex_I`, `sex_M`).
- El criterio cuantitativo exacto de filtrado de valores atípicos (eliminación de filas donde `height == 0`).

Nuestro registro en `data_log.json` permite entender las reglas aplicadas y, junto con el archivo original (`abalone_raw.csv`), el código `src/preprocess.py` y las dependencias fijadas en `requirements.txt`, alcanza para reproducir exactamente el dataset limpio.

### 2. ¿Qué pasaría si el dataset pesara varios GB o cambiara todas las semanas?

Guardar los datos directamente en Git sería problemático porque:
- El repositorio crecería demasiado (Git guarda copias de los objetos en el historial `.git`).
- Cada nueva versión semanal ocuparía espacio adicional.
- Clonar (`git clone`) y descargar el proyecto (`git pull`) sería cada vez más lento o superaría los límites de GitHub (100 MB por archivo).

Una alternativa sería separar el código de los datos, utilizando `.gitignore` para evitar almacenar los datasets directamente en Git y servicios externos como Amazon S3 o Google Cloud Storage para guardarlos. Además, podríamos utilizar herramientas como **DVC (Data Version Control)** para gestionar las versiones mediante archivos livianos de referencias y hashes, manteniendo la trazabilidad sin almacenar los archivos pesados en Git.

---
## Ejercicio 3

Esta etapa entrena un modelo reproducible para predecir la variable objetivo `age` a partir del conjunto de datos preprocesado (`data/processed/abalone_clean.csv`).

### 1. Requisitos previos
Asegúrate de tener instaladas las dependencias necesarias y de haber ejecutado previamente la etapa de preprocesamiento para contar con el archivo `data/processed/abalone_clean.csv`:

```bash
pip install pandas scikit-learn joblib
```

### 2. Entrenamiento y Evaluación del Modelo
Para entrenar el modelo, evaluar su desempeño en el conjunto de prueba y registrar los resultados, ejecuta el siguiente comando desde la raíz del proyecto:

```bash
python src/train_model.py
```

#### ¿Qué realiza este script?
* **División de datos:** Separa las características (`X`, excluyendo `age` y `rings`) y la variable objetivo (`y = age`), dividiendo los datos en entrenamiento (80%) y test (20%) con una semilla fija (`random_state=42`) para garantizar la reproducibilidad.
* **Pipeline de entrenamiento:** Construye un `Pipeline` que aplica escalado estándar (`StandardScaler`) ajustado únicamente con los datos de entrenamiento, seguido de un modelo de regresión de soporte vectorial (`SVR` con `kernel='rbf'`, `C=10.0`, `epsilon=0.2`, `gamma='scale'`).
* **Evaluación:** Calcula las métricas principales sobre el conjunto de prueba:
  * **MAE** (Error Absoluto Medio)
  * **RMSE** (Raíz del Error Cuadrático Medio)
  * **R²** (Coeficiente de Determinación)

### 3. Artefactos y Resultados Generados
Al finalizar la ejecución, se crean y actualizan automáticamente los siguientes archivos:

* **`models/model_v2.pkl`**: Pipeline completo entrenado y serializado con compresión (`compress=3`).
* **`results/metrics_v2.json`**: Archivo JSON con el detalle de la versión del modelo, hiperparámetros configurados, semilla utilizada y resultados de las métricas (MAE, RMSE, R²).
* **`results/experiments_log.csv`**: Registro histórico acumulativo de experimentos que almacena el ID del experimento (`experiment_id`), fecha y hora de ejecución (`timestamp`), versión del modelo, hiperparámetros y métricas obtenidas.

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

```bash
python src/train_model.py
```

### Experimento v3 — KNN (rama `exp/knn`)

Entrena un Pipeline (`StandardScaler` + `KNeighborsRegressor`). Los hiperparámetros (`n_neighbors`, `weights`, `p`) se eligen con `GridSearchCV` (KFold de 5 folds, `random_state=42`) usando sólo los datos de entrenamiento, y luego se evalúa sobre test. Genera `models/model_v3.pkl` y `results/metrics_v3.json`:

```bash
python src/train_knn.py
```

### Experimento v4 — Gradient Boosting (rama `exp/hiperparametros-iara`)

Entrena un Pipeline (`StandardScaler` + `GradientBoostingRegressor` con `n_estimators=100`, `learning_rate=0.1`, `random_state=42`). Genera `models/model_v4.pkl` y `results/metrics_v4.json`:

```bash
python src/train_model_gradb.py
```

### Experimento MLP — MLPRegressor (rama `exp/mlp-emasavat`)

Entrena un Pipeline (`StandardScaler` + `MLPRegressor`) con los hiperparámetros encontrados por
Optuna (ya fijados en el script). Genera `models/model_mlp_ema.pkl`,
`results/metrics_mlp_ema.json` y el gráfico `results/mlp_ema.png`:

```bash
python src/train_mlp_ema.py
```

**Opcional:** para repetir la búsqueda de hiperparámetros con Optuna (80 trials, puede tardar
varios minutos) y regenerar `results/optuna_trials_ema.csv`:

```bash
python src/optuna_mlp_ema.py
```

---

## Resultados

- **Modelo v1 (Random Forest - n_estimators=100, random_state=42):** MAE = 1.59 | RMSE = 2.25 | R² = 0.56
- **Modelo v2 (SVR - kernel='rbf', C=10.0, epsilon=0.2):** MAE = 1.51 | RMSE = 2.21 | R² = 0.57
- **Modelo v3 (KNN - n_neighbors=17, weights='distance', p=1):** MAE = 1.57 | RMSE = 2.33 | R² = 0.52
- **Modelo v4 (Gradient Boosting - n_estimators=100, learning_rate=0.1):** MAE = 1.56 | RMSE = 2.24 | R² = 0.56
- **Modelo MLP (MLPRegressor - (128, 64), Optuna):** MAE = 1.51 | RMSE = 2.15 | R² = 0.60

### Detalle MLPRegressor — Emanuel Savat

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

| Modelo | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest (v1 base) | 1.5880 | 2.2476 | 0.5573 |
| SVR (v2) | 1.5100 | 2.2100 | 0.5700 |
| KNN (v3) | 1.5700 | 2.3300 | 0.5200 |
| Gradient Boosting (v4) | 1.5600 | 2.2400 | 0.5600 |
| **MLPRegressor** | **1.5081** | **2.1462** | **0.5963** |

El modelo entrenado se encuentra en `models/model_mlp_ema.pkl`, las métricas en `results/metrics_mlp_ema.json` y el historial de Optuna en `results/optuna_trials_ema.csv`.

---

## Ejercicio 4 — Resolución de conflictos de Merge (NUEVO)

Al trabajar en ramas paralelas (`exp/algoritmo-gero`, `exp/knn`, `exp/hiperparametros-iara`, `exp/mlp-emasavat`) creadas a partir de `main`, cada integrante agregó la fila de su experimento al final de `results/experiments_log.csv` y una línea con sus métricas en la sección `## Resultados` de este `README.md`.

Al integrar la primera rama mediante Pull Request hacia `main`, la fusión fue automática. Sin embargo, al integrar las ramas siguientes surgieron conflictos (`CONFLICT (content)`), dado que distintas ramas habían modificado exactamente las mismas líneas finales de ambos archivos.

### Qué mostraban los archivos en conflicto

Al actualizar la rama con los cambios de `main`, Git insertó los marcadores de conflicto (`<<<<<<<`, `=======` y `>>>>>>>`) delimitando los cambios locales (`HEAD`) frente a los ya integrados en `main`:

**En `results/experiments_log.csv`:**
```text
<<<<<<< HEAD
exp_02,2026-10-25 19:15:30,v2,"{'kernel': 'rbf', 'C': 10.0, 'epsilon': 0.2}",1.51,2.21,0.57
=======
exp_03,2026-10-25 19:40:12,v3,"{'n_neighbors': 17, 'weights': 'distance', 'p': 1}",1.57,2.33,0.52
>>>>>>> main
```

**En `README.md` (sección Resultados):**
```text
<<<<<<< HEAD
**Modelo v2 (SVR - kernel='rbf', C=10.0, epsilon=0.2):** MAE = 1.51 | RMSE = 2.21 | R² = 0.57
=======
**Modelo v3 (KNN - n_neighbors=17, weights='distance', p=1):** MAE = 1.57 | RMSE = 2.33 | R² = 0.52
>>>>>>> main
```

### Criterio utilizado para resolverlo

Dado que tanto `results/experiments_log.csv` como la sección `## Resultados` del `README.md` son registros históricos acumulativos donde deben figurar todos los experimentos probados por el equipo, **el criterio de resolución fue conservar las líneas de ambas ramas** (ordenándolas secuencialmente una debajo de la otra y eliminando los marcadores `<<<<<<<`, `=======` y `>>>>>>>`). Asimismo, se mantuvieron scripts de entrenamiento separados en `src/` (`train_model.py`, `train_knn.py`, `train_model_gradb.py`, `train_mlp_ema.py`) para que todos los modelos puedan volver a ejecutarse sin pisarse entre sí.

---

## Ejercicio 5 — Verificación de la reproducibilidad

Para comprobar que el proyecto puede reproducirse desde cero, se clonó el repositorio en una carpeta nueva, recreando el entorno virtual `.venv` e instalando `requirements.txt`, y se ejecutaron el preprocesamiento y los entrenamientos siguiendo únicamente los pasos del `README.md`.

### Análisis de `git status` luego de re-ejecutar el flujo

1. **Archivos que aparecen como modificados y era esperable que cambiaran:**
   - `results/experiments_log.csv`: Aparece como `modified` porque cada script de entrenamiento abre el archivo en modo *append* (`mode="a"`) y escribe una nueva fila al final con la fecha y hora del momento de ejecución (`datetime.now()`).
   - `data/data_log.json` (si registra timestamp de corrida): Sólo varía el campo de fecha si se regenera dinámicamente.
   - `models/*.pkl`: Los archivos binarios comprimidos con `joblib` pueden presentar leves variaciones en bytes de metadatos según el sistema operativo o entorno, aunque el modelo entrenado sea exactamente el mismo.

2. **Archivos que confirman que el flujo se reproduce correctamente:**
   - `data/processed/abalone_clean.csv`: No presenta cambios en `git status` ni en `git diff`, demostrando que el preprocesamiento y la eliminación de atípicos (`height = 0`) son 100% deterministas.
   - `results/metrics_v2.json`, `metrics_v3.json`, `metrics_v4.json`, `metrics_mlp_ema.json`: Al re-entrenar los modelos, los valores de `MAE`, `RMSE` y `R2` coinciden exactamente con los versionados gracias a haber fijado `random_state=42` en `train_test_split` y en los estimadores.

### Acciones adicionales realizadas que no estaban inicialmente en el README

Durante la verificación en distintas computadoras surgieron tres situaciones prácticas que inicialmente no figuraban en el `README.md` y que luego fueron documentadas:
1. **Rutas demasiado largas en Windows (`Filename too long`):** Al clonar inicialmente dentro de directorios anidados de Windows sincronizados con OneDrive, Git arrojó un error por superar el límite de 260 caracteres (`MAX_PATH`). Fue necesario habilitar rutas largas con `git config --global core.longpaths true` o clonar en una ruta más corta.
2. **Política de ejecución de scripts en Windows PowerShell:** En PowerShell el comando `source` no existe y, al ejecutar `.venv\Scripts\Activate.ps1`, el sistema bloqueó la activación por permisos (`UnauthorizedAccess`). Fue necesario habilitar la política local con `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` (o activar mediante CMD con `activate.bat`).
3. **Bloqueo de consola por ventana de gráficos en `preprocess.py`:** Al correr `python src/preprocess.py`, `matplotlib` abre una ventana emergente con los boxplots y detiene la ejecución hasta que el usuario cierra dicha ventana manualmente, por lo que se sumó una nota aclaratoria en la sección de preprocesamiento.