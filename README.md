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

_(se completa en el Ejercicio 2)_

## Entrenamiento y evaluación

_(se completa en el Ejercicio 3)_

## Resultados

_(se completa en el Ejercicio 4)_
