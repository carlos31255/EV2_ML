# EP2 — Machine Learning (MLY1101)
## Caso C: Predicción de popularidad de canciones en Spotify

**Asignatura:** MLY1101 - Machine Learning  
**Evaluación:** Evaluación Parcial N°2 — Modelamiento  
**Institución:** Duoc UC  
**Dataset fuente:** [Spotify Tracks Dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) (HuggingFace)

---

## Descripción del proyecto

Este proyecto aplica la metodología **CRISP-DM** para construir un modelo de Machine Learning que predice si una canción en Spotify será un **hit comercial** (popularidad ≥ percentil 80, calculado solo con el conjunto de entrenamiento: 52 puntos tras deduplicar), usando únicamente sus características de audio y género musical.

**Pregunta de negocio:** *¿Debe un sello discográfico invertir presupuesto de promoción en esta canción?*

**KPIs de negocio:**

| KPI | Meta |
|---|---|
| AUC-ROC | ≥ 0.80 |
| Gap AUC train-test (control de sobreajuste) | < 0.05 |
| Recall | ≥ 0.75 |
| Lift @ 10% | ≥ 2.0 |

---

## Estructura del repositorio

```text
EV2_Machine_learning/
├── data/
│   ├── raw/
│   │   ├── Spotify_Tracks_Dataset.csv   # Dataset crudo descargado de HuggingFace
│   │   ├── train/raw_train.csv          # Generado por src/pipeline.py (deduplicado, sin transformar)
│   │   └── test/raw_test.csv
│   └── processed/
│       ├── train/                       # X_train.csv, y_train.csv, g_train.csv
│       ├── test/                        # X_test.csv, y_test.csv
│       └── feature_names.csv
├── docs/                                # Resumen de cada notebook (hallazgos, decisiones y resultados)
├── notebooks/
│   ├── 0_exploracion_inicial.ipynb      # Inspección inicial y EDA básico sobre datos crudos
│   ├── 1_EDA.ipynb                      # EDA profundo sobre train/test raw y análisis de sesgos
│   └── 2_Modelacion.ipynb               # Entrenamiento, evaluación y métricas
├── src/
│   ├── pipeline.py                      # Limpieza, deduplicación, multi-hot, split y transformaciones
│   ├── train.py                         # Validación cruzada y entrenamiento de los modelos
│   ├── test.py                          # Evaluación final en test y verificación de KPIs
│   ├── metricas.py                      # AUC, PR-AUC, precision, recall, F1, lift, gap, KPIs
│   └── graficos.py                      # Todas las visualizaciones (EDA y modelos)
├── output/
│   ├── graficos/                        # Gráficos generados
│   └── modelos/                         # Modelos (.joblib) y reportes de métricas (.csv)
├── main.py                              # Ejecuta el flujo completo
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

El entorno está configurado con las versiones compatibles y estables declaradas en el archivo `environment.yml`.

### 3. Registrar el kernel en Jupyter

```bash
python -m ipykernel install --user --name ml_spotify --display-name "Python (ml_spotify)"
```

### 4. Ejecutar el flujo completo

Desde la raíz del proyecto:

```bash
python main.py
```

Ejecuta, en este orden estricto: **pipeline → train → test → gráficos**.

| Paso | Módulo | Qué hace |
|---|---|---|
| 1 | `src/pipeline.py` | Descarta nulos, deduplica por `track_id`, codifica géneros (multi-hot), hace el split y aplica las transformaciones. |
| 2 | `src/train.py` | Entrena los 3 modelos con validación cruzada y elige el mejor por AUC de CV. |
| 3 | `src/test.py` | Evalúa en el conjunto de prueba y verifica los 4 KPIs de negocio. |
| 4 | `src/graficos.py` | Genera las visualizaciones en `output/graficos/`. |

Cada módulo también se puede ejecutar por separado: `python -m src.pipeline`, `python -m src.train`, `python -m src.test`.

### 5. Abrir los notebooks

Los notebooks leen los datos que genera el pipeline, así que se ejecutan **después** de `python main.py` (como mínimo, de `python -m src.pipeline`).

En VS Code / Antigravity IDE:
1. Abre el archivo del notebook (por ejemplo `notebooks/1_EDA.ipynb`).
2. En la esquina superior derecha, selecciona el kernel **"Python (ml_spotify)"** (o busca la ruta del intérprete en `~/anaconda3/envs/ml_spotify/python.exe`).
3. Ejecuta las celdas en orden.

O vía Jupyter Notebook / Lab tradicional:
```bash
jupyter notebook
```

| Orden | Notebook | Descripción |
|---|---|---|
| **1°** | `notebooks/0_exploracion_inicial.ipynb` | Inspección inicial y EDA básico sobre los datos crudos. |
| **2°** | `notebooks/1_EDA.ipynb` | EDA profundo con `raw_train.csv` y `raw_test.csv`: transformaciones, sesgo de género, popularidad 0, versiones y artistas. |
| **3°** | `notebooks/2_Modelacion.ipynb` | Entrenamiento, evaluación y métricas de los modelos. |

Todos los gráficos se guardan automáticamente en `output/graficos/` y los modelos y reportes en `output/modelos/`.

---

## Reproducibilidad desde cero

Antes de entregar, se borran los datos procesados, los splits crudos y los modelos, y se vuelve a ejecutar todo:

```bash
# Windows (PowerShell)
Remove-Item -Recurse -Force data\processed, data\raw\train, data\raw\test, output\modelos, output\graficos\*

# Linux / macOS
rm -rf data/processed data/raw/train data/raw/test output/modelos output/graficos/*

python main.py
```

El dataset original (`data/raw/Spotify_Tracks_Dataset.csv`) no se borra. Si no está en el repositorio, descárgalo desde HuggingFace y guárdalo con ese nombre en `data/raw/`.

---

## Alternativa sin Anaconda

### Opción A — Google Colab
Clona el repositorio dentro de Colab y ejecuta el flujo antes de abrir los notebooks:

```python
!git clone https://github.com/carlos31255/EV2_ML.git
%cd EV2_ML
!pip install -r requirements.txt
!python main.py
```

Luego abre los notebooks desde `notebooks/`:
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/carlos31255/EV2_ML/blob/main/notebooks/1_EDA.ipynb) **1 — EDA**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/carlos31255/EV2_ML/blob/main/notebooks/2_Modelacion.ipynb) **2 — Modelación**

### Opción B — Instalación vía pip
```bash
python -m venv .venv
# En Windows:
.venv\Scripts\activate
# En Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
python -m ipykernel install --user --name spotify_env --display-name "Python (spotify_env)"
python main.py
jupyter notebook
```

---

## Resultados principales (EV2 — Versión corregida)

Evaluación en conjunto de prueba con **18.034 canciones no vistas** (sin fuga de datos entre train y test).

Los resultados detallados de los modelos y la verificación de los KPIs se encuentran en el informe técnico y en los archivos generados durante la ejecución: `output/modelos/resultados_test.csv` y `output/modelos/kpis_cumplidos.csv`.

- **Umbral hit/no-hit:** popularidad ≥ 52 pts (percentil 80 calculado estrictamente sobre train y aplicado a test).
- **Deduplicación sin pérdida de géneros:** una fila por `track_id`, con la popularidad máxima entre copias y todos sus géneros consolidados.
- **Codificación de género:** multi-hot (`MultiLabelBinarizer`), que conserva el 18% de canciones multigénero que se perdería con un `drop_duplicates` directo.
- **Split por canción:** `GroupShuffleSplit` (80% train / 20% test, `random_state=42`) agrupado por `song_key` (artista + título), eliminando la fuga por canciones multigénero y por versiones.
- **Selección del modelo:** por AUC de validación cruzada agrupada (`StratifiedGroupKFold`), no por test.
- **Preprocesamiento integrado:** escalado y one-hot encapsulados en `Pipeline` de scikit-learn, de modo que se ajustan en cada fold.

---

## Limitaciones del modelo

- **El género domina la señal.** Usando solo el hit rate del género, el AUC es 0.837 en train y 0.831 en test, por encima del KPI de 0.80. Por eso el peso del género se mide con una ablación (el valor que devuelve `train.py` es AUC de validación cruzada con y sin género, no de test): cumplir el KPI no prueba por sí solo que las características de audio aporten.
- **Muestreo uniforme por género.** El dataset tiene la misma cantidad de filas por género, así que las tasas reflejan el muestreo y no el mercado real de Spotify; los resultados no se extrapolan sin cuidado.
- **Popularidad 0.** El 10.3% de las canciones tiene popularidad 0, concentrada en ciertos géneros (jazz, iranian, romance, soul, country, latin y rock superan el 50%). No se eliminan, sino que cuentan como "no hit"; pueden reflejar canciones sin dato más que canciones realmente impopulares.
- **Etiqueta ambigua en versiones.** Hay canciones que son hit en una versión y no hit en otra (1.061 combinaciones artista+título con versiones en ambas clases).
- **Split por canción, no por artista.** El 72.1% de las canciones de test tiene su artista también en train, y los artistas con varias canciones tienden a ser hit o no hit en bloque (63.4%), por lo que las métricas pueden ser algo optimistas.
- **Umbral relativo.** "Hit" se define como el 20% más popular del dataset (52 puntos en train), no como un criterio comercial absoluto.
- **`n_generos` excluida.** Se deja fuera de las features por ser un posible indicador indirecto de popularidad.

---

## Herramientas colaborativas

| Herramienta | Uso |
|---|---|
| **GitHub** | Control de versiones y repositorio central |
| **HuggingFace Hub** | Mirror público del dataset |
| **Google Colab** | Ejecución alternativa en la nube |
| **Discord** | Coordinación del equipo |
| **Anaconda** | Gestión de entornos reproducibles (`environment.yml`) |