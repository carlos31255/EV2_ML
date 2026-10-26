# EP2 — Machine Learning (MLY1101)
## Caso C: Predicción de popularidad de canciones en Spotify

**Asignatura:** MLY1101 - Machine Learning  
**Evaluación:** Evaluación Parcial N°2 — Modelamiento  
**Institución:** Duoc UC  
**Dataset fuente:** [Spotify Tracks Dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) (HuggingFace)

---

## Descripción del proyecto

Este proyecto aplica la metodología **CRISP-DM** para construir un modelo de Machine Learning que predice si una canción en Spotify será un **hit comercial** (popularidad ≥ percentil 80, calculado solo con el conjunto de entrenamiento: 52 puntos tras deduplicar), usando sus características de audio, su género musical y variables simples del texto (número de artistas y marcas del título, sin la identidad del artista).

**Pregunta de negocio:** *¿Debe un sello discográfico invertir presupuesto de promoción en esta canción?*

**Objetivo:** construir un modelo supervisado de clasificación binaria (hit = 1 / no hit = 0) que ordene el catálogo según su probabilidad de ser hit, para priorizar campañas de promoción, con un recall repartido de forma más pareja entre géneros y entre canciones de 1, 2 y 3+ géneros. Se comparan tres modelos supervisados (Regresión Logística, Random Forest y HistGradientBoosting) y se aplica K-Means como técnica no supervisada para segmentar canciones por su perfil de audio. El detalle (objetivos específicos, CRISP-DM e impacto ético) está en [`docs/informe_tecnico.md`](docs/informe_tecnico.md).

**KPIs de negocio:**

| KPI | Meta | Por qué |
|---|---|---|
| AUC-ROC | ≥ 0.80 | Corte habitual de buena capacidad para distinguir hits de no hits |
| Gap AUC train-test (control de sobreajuste) | < 0.05 | Regla práctica para detectar sobreajuste |
| Recall | ≥ 0.75 | Prioriza no perder hits: perder uno cuesta más que promocionar una canción que no lo es |
| Lift @ 10% | ≥ 2.0 | El 10% superior debe tener al menos el doble de hits que una selección al azar |

Son criterios definidos por el equipo, no un estándar externo. No incluyen una meta de precisión (ver limitaciones).

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
│   └── 2_Modelacion.ipynb               # Entrenamiento, evaluación, política de umbral, K-Means y sesgos
├── src/
│   ├── pipeline.py                      # Limpieza, deduplicación, multi-hot, split y transformaciones
│   ├── train.py                         # Validación cruzada, entrenamiento y calibración de la política
│   ├── test.py                          # Evaluación final en test, KPIs y equidad
│   ├── metricas.py                      # AUC, PR-AUC, precision, recall, F1, lift, gap, KPIs y política de umbral
│   └── graficos.py                      # Todas las visualizaciones (EDA y modelos)
├── output/
│   ├── graficos/                        # Gráficos generados
│   └── modelos/                         # Modelos y política de umbral (.joblib)
│       └── resultados/                  # Tablas de resultados (.csv)
├── main.py                              # Ejecuta el flujo completo
├── environment.yml                      # Entorno conda reproducible (versiones fijadas)
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
| 1 | `src/pipeline.py` | Descarta nulos, deduplica por `track_id`, codifica géneros (multi-hot), hace el split y aplica las transformaciones (`log1p` y variables de texto). |
| 2 | `src/train.py` | Entrena los 3 modelos con validación cruzada, elige el mejor por AUC de CV y calibra la política de umbral con probabilidades fuera de muestra. |
| 3 | `src/test.py` | Evalúa en el conjunto de prueba (los 3 modelos con umbral 0.5 y el modelo oficial con su política), verifica los 4 KPIs y mide la equidad por género. |
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

Todos los gráficos se guardan automáticamente en `output/graficos/`, los modelos en `output/modelos/` y las tablas de resultados en `output/modelos/resultados/`.

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

`output/graficos/*` borra todos los gráficos, incluidos los del K-Means ampliado (`17_kmeans_criterios_k.png`, `18_kmeans_pca_clusters.png`, `19_kmeans_generos.png`) y el de la política de umbral (`20_politica_umbral.png`). `main.py` regenera solo los gráficos 01 a 12; el 00 lo genera `0_exploracion_inicial.ipynb` y del 13 al 20 los genera `2_Modelacion.ipynb`. Por eso, después de `python main.py`, hay que ejecutar los tres notebooks en orden con **Restart & Run All**. La grilla de hiperparámetros de la sección 11 tarda ≈ 25 minutos la primera vez; si `output/modelos/resultados/busqueda_hiperparametros.csv` existe, se reutiliza (como este comando borra `output/modelos`, se vuelve a calcular, igual que la comparación de `class_weight` de la sección 2b).

**Qué es determinista y qué no.** El split y el umbral P80 se reproducen exactamente, y también el AUC, el gap y el lift de test. Con las versiones fijadas en `requirements.txt` / `environment.yml` (scikit-learn 1.8.0, pandas 3.0.3, numpy 2.4.6) los resultados se reproducen exactamente; se verificó con corridas idénticas en Windows y Linux. Con otra versión de scikit-learn cambian los resultados que dependen de la validación cruzada (por ejemplo, con 1.5.2 el recall oficial da 0.795), así que conviene comprobar las versiones antes de ejecutar. Para verificarlas: `python -c "import sklearn, pandas, numpy; print(sklearn.__version__, pandas.__version__, numpy.__version__)"`. Si un entorno `ml_spotify` ya existía con otras versiones, actualízalo con `conda env update -n ml_spotify -f environment.yml --prune`.

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

## Resultados principales

Evaluación en conjunto de prueba con **18.034 canciones no vistas** (sin fuga de datos entre train y test).

**Modelo oficial:** HistGradientBoosting (profundidad libre, 31 hojas, `min_samples_leaf=20`, L2 = 1) con variables de audio, género y texto, más una **política de umbral por género y número de géneros** calibrada en train (`src/metricas.py`). Cumple los cuatro KPIs: AUC 0.866, gap 0.030, recall 0.791 y lift@10% 3.43. Frente a umbral 0.5, la política sube el recall en géneros de nicho de 0.14 a 0.35 e iguala el recall entre canciones de 1, 2 y 3+ géneros (≈ 0.79; las de 3+ bajan de 2.18 a 1.74 veces su tasa de hits, pero las de 1 género suben de 1.82 a 2.02). Cuesta ≈ 3 puntos de precisión frente al umbral 0.5 (0.458 → 0.423) y ≈ 6 frente a un umbral único con el mismo recall (0.483).

El detalle está en el informe técnico y en los archivos generados durante la ejecución: `output/modelos/resultados/resultados_test.csv`, `output/modelos/resultados/kpis_cumplidos.csv` y `output/modelos/resultados/equidad_test.csv`.

- **Umbral hit/no-hit:** popularidad ≥ 52 pts (percentil 80 calculado estrictamente sobre train y aplicado a test).
- **Deduplicación sin pérdida de géneros:** una fila por `track_id`, con la popularidad máxima entre copias y todos sus géneros consolidados.
- **Codificación de género:** multi-hot (`MultiLabelBinarizer`), que conserva el 18% de canciones multigénero que se perdería con un `drop_duplicates` directo.
- **Split por canción:** `GroupShuffleSplit` (80% train / 20% test, `random_state=42`) agrupado por `song_key` (artista + título), eliminando la fuga por canciones multigénero y por versiones.
- **Selección del modelo:** por AUC de validación cruzada agrupada (`StratifiedGroupKFold`), no por test. La política de umbral se calibra con probabilidades fuera de muestra de train.
- **Preprocesamiento integrado:** escalado y one-hot encapsulados en `Pipeline` de scikit-learn, de modo que se ajustan en cada fold.

---

## Limitaciones del modelo

- **El género domina la señal.** Usando solo el hit rate del género, el AUC es 0.837 en train y 0.831 en test, por encima del KPI de 0.80. Por eso el peso del género se mide con una ablación (el valor que devuelve `train.py` es AUC de validación cruzada con y sin género, no de test): cumplir el KPI no prueba por sí solo que las características de audio aporten. Sin género, el AUC de CV cae de 0.868 a 0.707.
- **Sesgo entre géneros reducido, no eliminado.** La política de umbral sube el recall en géneros de nicho de 0.14 a 0.35, pero el 51% de los géneros sigue bajo el KPI de 0.75, y el 10% superior del ranking (la campaña recomendada) sigue concentrado en géneros populares. Ordenar por margen sobre el umbral del género reparte mejor la campaña (lift 2.44); queda como alternativa.
- **Trato distinto según el género.** Con la política, dos canciones con la misma probabilidad pueden recibir decisiones distintas según su género: es una decisión de negocio explícita.
- **Muestreo uniforme por género.** El dataset tiene la misma cantidad de filas por género, así que las tasas reflejan el muestreo y no el mercado real de Spotify; los resultados no se extrapolan sin cuidado.
- **Popularidad 0.** El 10.3% de las canciones de train tiene popularidad 0, concentrada en ciertos géneros (jazz, iranian, romance, soul, country, latin y rock superan el 50%). No se eliminan, sino que cuentan como "no hit"; pueden reflejar canciones sin dato más que canciones realmente impopulares.
- **Etiqueta ambigua en versiones.** Hay canciones que son hit en una versión y no hit en otra (1.061 combinaciones artista+título con versiones en ambas clases).
- **Split por canción, no por artista.** El 72.1% de las canciones de test tiene su artista también en train, y los artistas con varias canciones tienden a ser hit o no hit en bloque (63.4%), por lo que las métricas pueden ser algo optimistas. Con un split por artista principal el AUC baja ≈ 0.008, pero es una cota inferior: el 36% del test sigue teniendo artistas vistos por colaboraciones.
- **Umbral relativo.** "Hit" se define como el 20% más popular del dataset (52 puntos en train), no como un criterio comercial absoluto.
- **`n_generos` excluida.** Se deja fuera de las features por ser un posible indicador indirecto de popularidad. El multi-hot lo lleva implícito: el AUC es menor en canciones multigénero, y la política de umbral iguala su recall con el de las de 1 género.
- **KPIs sin meta de precisión.** El modelo cumple los cuatro KPIs aunque el 58% de lo que marca no es hit. Los parámetros de la política (recall objetivo 0.78, encogimiento 30) y la regla de hiperparámetros (gap de CV < 0.04) son decisiones de diseño sin análisis de sensibilidad.
- **Random Forest** no cumple el KPI de recall con umbral 0.5 (0.686); no es el modelo elegido.

---

## Herramientas colaborativas

| Herramienta | Uso |
|---|---|
| **GitHub** | Control de versiones y repositorio central |
| **HuggingFace Hub** | Mirror público del dataset |
| **Google Colab** | Ejecución alternativa en la nube |
| **Discord** | Coordinación del equipo |
| **Anaconda** | Gestión de entornos reproducibles (`environment.yml`) |