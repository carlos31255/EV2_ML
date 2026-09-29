# EP2 — Machine Learning (MLY1101)
## Caso C: Predicción de popularidad de canciones en Spotify

**Asignatura:** MLY1101 - Machine Learning  
**Evaluación:** Evaluación Parcial N°2 — Modelamiento  
**Institución:** Duoc UC  
**Dataset fuente:** [Spotify Tracks Dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) (HuggingFace)

---

## Descripción del proyecto

Este proyecto aplica la metodología **CRISP-DM** para construir un modelo de Machine Learning que predice si una canción en Spotify será un **hit comercial** (popularidad ≥ percentil 80 = 54 puntos), usando únicamente sus características de audio y género musical.

**Pregunta de negocio:** *¿Debe un sello discográfico invertir presupuesto de promoción en esta canción?*

---

## Estructura del repositorio

```text
EV2_Machine_learning/
├── data/
│   ├── raw/
│   │   └── Spotify_Tracks_Dataset.csv   # Dataset crudo descargado de HuggingFace
│   └── processed/
│       └── spotify_clean.csv            # Dataset limpio generado por EV1 (113.549 filas)
├── docs/                                # Pauta y documentación de la evaluación
├── images/                              # Gráficos y matrices de confusión generados
├── models/                              # Modelos entrenados serializados (.joblib)
│   ├── best_model.joblib                # Mejor modelo (HistGradientBoosting / Random Forest)
│   ├── gradient_boosting.joblib         # Pipeline HistGradientBoosting
│   ├── random_forest.joblib             # Pipeline Random Forest
│   └── logistic_regression.joblib       # Pipeline Regresión Logística
├── notebooks/
│   ├── EV1_Spotify.ipynb               # EDA + Limpieza + Preparación de datos
│   └── EV2_Spotify_Modelos.ipynb       # Modelamiento + Evaluación + Guardado de modelos
├── environment.yml                      # Entorno conda reproducible (numpy 1.26 + pandas 2.2)
├── requirements.txt                     # Alternativa pip
├── notas_EV2_Spotify.md                 # Bitácora técnica y registro de decisiones
└── README.md                            # Este archivo
```

---

## Requisitos previos

- [Git](https://git-scm.com/)
- [Anaconda](https://www.anaconda.com/download) o [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

---

## Reproducción paso a paso

### 1. Clonar el repositorio

```bash
git clone https://github.com/carlos31255/EV2_ML.git
cd EV2_ML
```

### 2. Crear y activar el entorno conda

```bash
conda env create -f environment.yml
conda activate ml_spotify
```

El entorno está configurado con versiones compatibles y estables para Windows:
- **Python:** 3.11
- **NumPy:** 1.26.4
- **Pandas:** 2.2.2
- **Scikit-Learn:** 1.5.2
- **Matplotlib:** 3.9.2
- **Seaborn:** 0.13.2
- **python-pptx:** 1.0.2

### 3. Registrar el kernel en Jupyter

```bash
python -m ipykernel install --user --name ml_spotify --display-name "Python (ml_spotify)"
```

### 4. Abrir y ejecutar los notebooks

En VS Code / Antigravity IDE:
1. Abre el archivo del notebook (`notebooks/EV2_Spotify_Modelos.ipynb`).
2. En la esquina superior derecha, selecciona el kernel **"Python (ml_spotify)"** (o busca la ruta del intérprete en `~/anaconda3/envs/ml_spotify/python.exe`).
3. Ejecuta las celdas en orden.

O vía Jupyter Notebook / Lab tradicional:
```bash
jupyter notebook
```

| Orden | Notebook | Descripción |
|---|---|---|
| **1°** | `notebooks/EV1_Spotify.ipynb` | Descarga el dataset desde HuggingFace (automático si no existe localmente), realiza el EDA completo, limpieza y genera `data/processed/spotify_clean.csv`. |
| **2°** | `notebooks/EV2_Spotify_Modelos.ipynb` | Lee el dataset limpio, aplica el preprocesamiento corregido y entrena los 3 modelos de clasificación. |

Todos los gráficos se guardan automáticamente en `images/` y los datasets procesados en `data/processed/`.

---

## Alternativa sin Anaconda

### Opción A — Google Colab
Puedes abrir directamente los cuadernos en Google Colab:
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/carlos31255/EV1_Machine_learning/blob/main/notebooks/EV1_Spotify.ipynb) **EV1 — EDA y Limpieza**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/carlos31255/EV2_ML/blob/main/notebooks/EV2_Spotify_Modelos.ipynb) **EV2 — Modelamiento y Evaluación**

### Opción B — Instalación vía pip
```bash
python -m venv .venv
# En Windows:
.venv\Scriptsctivate
# En Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
python -m ipykernel install --user --name spotify_env --display-name "Python (spotify_env)"
jupyter notebook
```

---

## Resultados principales (EV2 — Versión corregida)

Evaluación en conjunto de prueba con **18.034 canciones no vistas** (sin fuga de datos entre train y test):

| Modelo | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Observación |
|---|---|---|---|---|---|---|
| Regresión Logística | 0.7338 | 0.4216 | 0.7979 | 0.5517 | 0.8357 | Baseline interpretable, alto recall |
| Random Forest | 0.7283 | 0.4126 | 0.7631 | 0.5356 | 0.8241 | Ensamble robusto, sin sobreajuste (gap 0.02) |
| **HistGradientBoosting** | **0.7327** | **0.4228** | **0.8271** | **0.5595** | **0.8544** | ⭐ **Mejor modelo para el negocio** (mayor AUC, F1 y recall) |

- **Umbral hit/no-hit:** popularidad ≥ 52 pts (Percentil 80 calculado estrictamente sobre train).
- **Split por canción:** `GroupShuffleSplit` (80% train / 20% test) agrupado por `song_key`, eliminando la fuga por canciones multigénero.
- **Preprocesamiento integrado:** escalado y codificación multi-hot/one-hot encapsulados en `Pipeline` de scikit-learn.

---

## Herramientas colaborativas

| Herramienta | Uso |
|---|---|
| **GitHub** | Control de versiones y repositorio central |
| **HuggingFace Hub** | Mirror público del dataset (descarga automática sin credenciales) |
| **Google Colab** | Ejecución alternativa en la nube |
| **Discord** | Coordinación del equipo |
| **Anaconda** | Gestión de entornos reproducibles (`environment.yml`) |
