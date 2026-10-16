# EV2 - Machine Learning - Spotify Tracks (Resumen Ejecutivo)

## 1. Definición del Problema y Objetivos
- **Problema:** Predecir si una canción será un "hit" (popularidad >= P80) para optimizar campañas de marketing.
- **Objetivo:** Modelo supervisado de clasificación binaria (hit = 1 o 0).
- **KPIs de Negocio:**
  - AUC-ROC >= 0.80.
  - Gap AUC train-test < 0.05 (control de sobreajuste).
  - Recall >= 0.75.
  - Lift @ 10% >= 2.0.

## 2. Flujo de Trabajo y Decisiones Técnicas
1. **EDA Inicial (Raw Data):** El análisis exploratorio debe hacerse al inicio sobre los datos crudos para entender distribuciones y detectar sesgos antes de alterar la data. Se debe comentar en 1_EDA.ipynb que las canciones con Popularity = 0 no se eliminan y se analiza su efecto (no son errores, concentran ciertos artistas/géneros).
2. **Preprocesamiento y Mitigación de Sesgos:**
   - **Nulos:** Descartar la fila con nulos de texto (es 1 fila; los nulos están en artists, album_name y track_name; no se imputan, no aportan al modelo).
   - **Deduplicación Agrupada:** No usar drop_duplicates directo para no perder géneros. Se agrupa por track_id, la popularidad es el máximo entre copias y se consolida la lista de todos sus géneros.
   - **Codificación de Género (track_genre):** Al tener múltiples géneros por canción, se usa **multi-hot encoding (MultiLabelBinarizer)**. Conserva el 18% de información que se perdería. *(Reportar AUC con y sin género para medir su peso en el hit.)*
   - **Escalado:** Numéricas según distribución (StandardScaler tras aplicar log1p en asimétricas), dentro del Pipeline.
3. **Splits (Train/Test):**
   - El split se hace **después de deduplicar, pero ANTES de transformaciones** para guardar tanto raw como processed.
   - Se agrupa por canción única (song_key = artists + track_name) con GroupShuffleSplit (evita que la misma canción caiga en train y test inflando el AUC).
   - Semilla estricta (random_state=42).
   - El umbral **P80 se calcula SOLO con train** (52.0) y se aplica a test (el P80 cambia de 54 a ~52 tras deduplicar).
4. **Modelos:** Regresión Logística, Random Forest, HistGradientBoosting (supervisados) y K-Means (no supervisado). El oficial es HGB ajustado + variables de texto + política de umbral por género y n° de géneros (`src/metricas.py`).

## 3. Estructura Oficial
- data/raw/ y data/processed/ (ambos con train/ y test/).
- notebooks/:
  - 0_exploracion_inicial.ipynb (inspección inicial y EDA básico en raw).
  - 1_EDA.ipynb (visualizaciones profundas y análisis de sesgos, como el efecto de popularity 0 y el peso del género).
  - 2_Modelacion.ipynb (entrenamiento, evaluación, análisis de sesgos y pruebas de mitigación).
- src/: pipeline.py, train.py, test.py, metricas.py (incluye la política de umbral), graficos.py.
- output/: graficos/ y modelos/. Gráficos nuevos del K-Means ampliado: 17_kmeans_criterios_k, 18_kmeans_pca_clusters, 19_kmeans_generos.
- docs/: 0_exploracion_inicial.md, 1_eda.md, 2_modelo.md, informe_tecnico.md, Explicacion_Variables_Spotify.pdf.
- main.py y README.md.

## 4. Estado (Octubre 2026)
- Lógica de MultiLabelBinarizer, deduplicación agregada y split seguro por song_key restaurados.
- **Modelo oficial actualizado (14-10-2026):** HistGradientBoosting con hiperparámetros elegidos por CV (profundidad libre, 31 hojas, `min_samples_leaf=20`, L2 = 1), 5 variables de texto sin identidad del artista y política de umbral por género + número de géneros calibrada con probabilidades fuera de muestra de train. Multi-hot binario, split por canción, target absoluto (P80 de train = 52), popularidad 0 como no hit.
- `main.py` y los tres notebooks ejecutados desde cero en Linux, sin errores (2_Modelacion: 34 de 34 celdas, en orden). Textos de lectura alineados con las salidas.

## 5. Ajustes de Etapas Anteriores (Verificados y Cumplidos al 100%)
1. **requirements.txt (Cumplido):** Incluye `scipy==1.17.1` y `joblib==1.5.3` fijados (junto a pandas, numpy, scikit-learn, matplotlib, seaborn, jupyter, ipykernel y python-pptx).
2. **src/test.py (Cumplido):** Implementa el chequeo preventivo `_hay_modelos()` ("ejecuta python main.py o python -m src.train primero") retornando `{}` limpiamente sin romper, y concentra toda la evaluación final de los 4 KPIs guardando `kpis_cumplidos.csv` y `resultados_test.csv`.
3. **main.py (Cumplido):** Flujo estrictamente ordenado `pipeline -> train -> test -> graficos` (paso 4 activo con `try/except` para no interrumpir la ejecución).
4. **src/graficos.py (Cumplido):** Concentra 29 funciones de visualización (gráficos 00 al 16, tablas y `run()`); los tres notebooks importan de forma centralizada vía `from src import graficos as gr`.
5. **docs/ (Cumplido):** Redactados los tres .md por notebook (`0_exploracion_inicial.md`, `1_eda.md`, `2_modelo.md`), el informe técnico (`informe_tecnico.md`) y el documento técnico en PDF (`Explicacion_Variables_Spotify.pdf`).

## 6. Resultados Clave de 2_Modelacion (corrida final, semilla 42, Linux)
- **Modelo oficial (test):** AUC 0.866, gap 0.030, recall 0.791, precisión 0.423, lift@10% 3.43. Cumple los 4 KPIs. Con umbral 0.5 el mismo modelo da recall 0.824 y precisión 0.458. Random Forest no cumple el KPI de recall con 0.5 (0.686).
- **Frente a la versión anterior** (HGB `max_depth=6`, sin texto, umbral 0.5: AUC 0.857, recall 0.833, precisión 0.425, lift 3.36): sube AUC y lift, la precisión se mantiene, el recall en géneros de nicho pasa de 0.148 a 0.354 y el sobre-marcado de canciones con 3+ géneros baja de 2.27 a 1.74 veces su hit rate.
- **Género:** sin género el AUC de CV cae de 0.868 a 0.707; por permutación el género pesa 0.31 de AUC (la siguiente variable, 0.009). Con la política: recall macro por género 0.559 → 0.591, correlación hit rate-recall 0.95 → 0.83, pero el 51% de los géneros sigue bajo 0.75.
- **Multigénero:** marcadas / hit rate (1 · 2 · 3+ géneros) pasa de 1.82 · 1.59 · 2.18 (umbral 0.5) a 2.02 · 1.43 · 1.74; recall parejo ≈ 0.79 en los tres grupos.
- **Bootstrap (300 réplicas):** frente a 0.5, recall de nicho +0.21 [0.15; 0.26], precisión −0.03 [−0.04; −0.03].
- **Campaña del 10% superior:** sigue sesgada (1.5% de los hits de nicho, 68 de 114 géneros). Ordenando por margen sobre el umbral del género: 14.4% y 91 géneros, lift 2.44.
- **K-Means (audio):** k=2 por silhouette (0.240) separa solo "tranquilo" de "enérgico". Con k=7 (estable, ARI >= 0.957 entre mitades) los clusters son familias de estilo con hit rate entre ≈ 0.10 y ≈ 0.24; usar solo el cluster da AUC 0.576. Los géneros agrupados por audio (k=3) tienen acuerdo moderado entre K-Means y Ward (ARI 0.47); un grupo de 12 géneros acústicos/instrumentales tiene hit rate medio 0.12 frente a 0.245 de los otros.
- **Hiperparámetros (sección 11):** grilla de 32 configuraciones en la meseta `min_samples_leaf=20`, regla "mayor AUC CV con gap CV < 0.04". La oficial empata en la cima (0.8661 vs 0.8663, diferencia menor que la desviación entre folds); la anterior queda 24ª de 33.
- **Número de géneros (sección 12):** solo con canciones de un género el AUC es 0.875 (oficial en las mismas canciones, 0.875).
- **Variables de texto (sección 13):** +0.003 de AUC en CV; el ensamble con Regresión Logística empeora (test 0.858).
- **Negocio (sección 14):** el 10% superior del ranking tiene ≈ 70% de hits y captura ≈ 34% de todos los hits; la política oficial marca el 38% del catálogo y el 58% de lo marcado no es hit.

## 7. Decisión Final y Alternativas Documentadas
- Se adopta el modelo oficial descrito en la sección 4.
- Alternativas documentadas, sin adoptar: equidad fuerte (umbral por género sin encogimiento, recall 0.82: nicho 0.71 pero precisión 0.285), ranking por margen para la campaña (lift 2.44), modelo con una fila por (canción, género) (AUC 0.849, empeora el sobre-marcado), reponderar por género (AUC 0.686), ensamble con Regresión Logística.
- El sesgo de género queda reducido y medido, no eliminado. El modelo captura sobre todo qué géneros son populares, no qué hace "buena" a una canción.
- Multi-hot normalizado: evaluado en una corrida previa con el modelo anterior y descartado; no tiene celda en el notebook.

## 8. Rúbrica EV2 y Defensa
- Evaluación grupal con defensa individual: 10 minutos de presentación y preguntas cruzadas (cada integrante debe dominar todo el proyecto).
- Entrega: informe técnico .md reproducible tipo README (problema, objetivos, KPIs, fuentes de datos, EDA, CRISP-DM), notebook .ipynb ejecutable, datos y archivos complementarios, carpeta con estructura profesional.
- Requisitos: mínimo 2 modelos supervisados y 1 técnica no supervisada; evaluar el impacto ético.
- Ponderación: IE5 metodología 20%, IE6 modelos supervisados (algoritmo y configuración justificados) 30%, IE7 no supervisado (patrones relevantes y aplicación al negocio) 30%, IE8 interpretación de métricas y recomendaciones de negocio 20%.
- Preguntas probables: por qué no se quitó el género, por qué un umbral distinto por género (y si eso es justo), cómo se calibró la política sin mirar test, por qué HGB y esa configuración, qué significa silhouette 0.24 y por qué se segmenta con k=7, qué mide el lift, por qué split por canción, qué mide el gap.

## 9. Pendientes Antes de Entregar
1. ~~informe_tecnico.md: secciones de Objetivos, Impacto ético y mapeo CRISP-DM.~~ Hecho (secciones 2, 5 y 10 del informe).
2. ~~Confirmar la fuente de datos.~~ Hecho: las 114.000 filas de `data/raw/Spotify_Tracks_Dataset.csv` son idénticas a las del CSV entregado por la asignatura (solo cambia el nombre de la columna índice). No se pudo descargar el archivo de HuggingFace para compararlo directamente.
3. ~~Multi-hot normalizado en docs.~~ Hecho: marcado como "evaluado en una corrida previa" en 1_eda.md y 2_modelo.md.
4. ~~README.md.~~ Hecho: objetivo agregado, comentario de environment.yml corregido, nota sobre gráficos 00 y 13 a 19 (incluye 17 a 19) en la limpieza.
5. Reproducibilidad: verificada en Linux (main.py y los tres notebooks sin errores, 34 de 34 celdas en 2_Modelacion). Falta repetir `python main.py` y "Restart & Run All" en Windows antes de entregar; el recall y la precisión oficiales pueden moverse en el tercer decimal (dependen de la CV).
6. Ensayar la defensa con las preguntas de la sección 8.

## 10. Registro de Cambios (Octubre 2026)
- Se actualizó el texto sobre la selección del mejor modelo en la última celda de `notebooks/2_Modelacion.ipynb`.
- Se actualizó la tabla de resultados en `README.md` con los valores de `output/modelos/resultados_test.csv`.
- Se clarificó en `README.md` que la ablación de género se mide con AUC de validación cruzada.
- Se eliminó la tabla de resultados de `README.md` apuntando al informe técnico y archivos CSV.
- Se eliminó la lista de versiones codificada en `README.md` para evitar desfase con `environment.yml`.
- Se borraron los 4 gráficos viejos (`08_roc_pr.png`, `09_recall_por_genero.png`, `10_importancia_permutacion.png`, `11_kmeans_silhouette.png`) y se añadió su ruta a los comandos de limpieza del `README.md`.
- Se actualizaron los tamaños de los conjuntos train (71.706) y test (18.034) en `README.md` y `eda.md` para reflejar el reentrenamiento.
- Se creó y redactó el informe técnico consolidado en `docs/informe_tecnico.md`.
- Se movió `eda.md` a `docs/1_eda.md` y se redactaron los resúmenes faltantes (`docs/0_exploracion_inicial.md` y `docs/2_modelacion.md`) para cumplir estrictamente con la estructura de la evaluación.
- Se actualizó `docs/informe_tecnico.md` corrigiendo la mención de Recall a 0.833 en la sección 7.3 e incorporando el detalle de disparidad del 51% de géneros bajo el KPI.
- Se actualizó `docs/2_modelo.md` reflejando el K-Means ampliado (k=7) y documentando las pruebas de mitigación de la sección 10b (política suavizada por género) e hiperparámetros de la sección 11 añadidas a `notebooks/2_Modelacion.ipynb`.
- Se generó el documento técnico en PDF `docs/Explicacion_Variables_Spotify.pdf` detallando las variables descartadas, transformadas y mantenidas, junto a la justificación técnica, arquitectura de datos y relación con los 4 KPIs.
- Se actualizaron los textos explicativos en `notebooks/2_Modelacion.ipynb` (secciones 10, 11, 13 y resumen final) y `docs/2_modelo.md` tras verificar que las salidas numéricas correspondientes a la política suavizada (λ=0,1), búsqueda de hiperparámetros (9ª posición) y variables de texto coinciden con la ejecución.
- Se ejecutó `notebooks/2_Modelacion.ipynb` completo (Restart & Run All): 34 de 34 celdas en orden y sin errores.
- Se reorganizaron las notas: estado actual, resultados clave, decisión final, rúbrica de la EV2 y pendientes antes de entregar.
- Se verificó y registró el cumplimiento al 100% de los 5 ajustes de etapas anteriores (requirements.txt, src/test.py, main.py, src/graficos.py y docs/).
- Se completó `docs/informe_tecnico.md`: objetivos del proyecto, fuente de datos verificada, tabla CRISP-DM, K-Means k=7 y su uso de negocio, lectura de cada KPI, umbral corregido (antes decía que 0.5 maximiza el recall), recomendaciones de negocio, alternativas evaluadas, impacto ético y reproducción.
- Corrida desde cero en Linux con el modelo anterior: las métricas de test eran idénticas a las de Windows; los resultados que dependen de CV variaban en el tercer decimal por diferencias numéricas entre sistemas operativos en el entrenamiento (no por los datos: con los mismos archivos de Windows el AUC de CV en Linux da igual 0.8568).
- **Modelo mejorado (14-10-2026):** `src/pipeline.py` agrega 5 variables de texto; `src/train.py` usa HGB con profundidad libre, 31 hojas, `min_samples_leaf=20`, L2 = 1, Regresión Logística con `lbfgs` (mismo AUC, ≈ 17 veces más rápida) y calibra la política de umbral (funciones en `src/metricas.py`, guarda `politica_umbral.joblib` y `prob_oof_train.csv`); `src/test.py` evalúa el modelo oficial con la política y guarda `equidad_test.csv`; `src/graficos.py` dibuja las curvas una vez por modelo.
- `notebooks/2_Modelacion.ipynb`: secciones 1, 2, 4, 4b, 6e, 10 (nueva política, bootstrap y campaña del 10%), 11 (grilla de 32 configuraciones, cacheada en `busqueda_hiperparametros.csv`), 12, 13 y 14 actualizadas; lecturas reescritas con las salidas nuevas.
- Docs actualizados: `informe_tecnico.md`, `2_modelo.md`, `1_eda.md`, `README.md` y `Explicacion_Variables_Spotify.pdf` (133 variables, nuevos KPIs).
- **Estructura alineada con la pizarra (15-10-2026):** la política de umbral se movió de `src/umbral.py` a `src/metricas.py`, así `src/` queda con pipeline, graficos, metricas, train y test. Nuevo gráfico `20_politica_umbral.png` (función `grafico_politica_umbral` en `src/graficos.py`, sección 10b del notebook 2). Los gráficos de evidencia (00 a 20) quedaron insertos en `0_exploracion_inicial.md`, `1_eda.md`, `2_modelo.md` e `informe_tecnico.md`.
