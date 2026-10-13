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
4. **Modelos:** Regresión Logística, Random Forest, HistGradientBoosting (supervisados) y K-Means (no supervisado).

## 3. Estructura Oficial
- data/raw/ y data/processed/ (ambos con train/ y test/).
- notebooks/:
  - 0_exploracion_inicial.ipynb (inspección inicial y EDA básico en raw).
  - 1_EDA.ipynb (visualizaciones profundas y análisis de sesgos, como el efecto de popularity 0 y el peso del género).
  - 2_Modelacion.ipynb (entrenamiento, evaluación, análisis de sesgos y pruebas de mitigación).
- src/: pipeline.py, train.py, test.py, metricas.py, graficos.py.
- output/: graficos/ y modelos/. Gráficos nuevos del K-Means ampliado: 17_kmeans_criterios_k, 18_kmeans_pca_clusters, 19_kmeans_generos.
- docs/: 0_exploracion_inicial.md, 1_eda.md, 2_modelo.md, informe_tecnico.md, Explicacion_Variables_Spotify.pdf.
- main.py y README.md.

## 4. Estado (Octubre 2026)
- Lógica de MultiLabelBinarizer, deduplicación agregada y split seguro por song_key restaurados.
- 2_Modelacion.ipynb ampliado (K-Means con k=7, secciones 10 a 14) y ejecutado completo de arriba a abajo: 34 de 34 celdas, en orden, sin errores. Textos de lectura alineados con las salidas.
- El modelo oficial no cambia (no se modificó train.py): HistGradientBoosting, multi-hot binario, split por canción, target absoluto (P80 de train = 52), popularidad 0 como no hit, umbral 0.5.

## 5. Ajustes de Etapas Anteriores (Verificados y Cumplidos al 100%)
1. **requirements.txt (Cumplido):** Incluye `scipy==1.17.1` y `joblib==1.5.3` fijados (junto a pandas, numpy, scikit-learn, matplotlib, seaborn, jupyter, ipykernel y python-pptx).
2. **src/test.py (Cumplido):** Implementa el chequeo preventivo `_hay_modelos()` ("ejecuta python main.py o python -m src.train primero") retornando `{}` limpiamente sin romper, y concentra toda la evaluación final de los 4 KPIs guardando `kpis_cumplidos.csv` y `resultados_test.csv`.
3. **main.py (Cumplido):** Flujo estrictamente ordenado `pipeline -> train -> test -> graficos` (paso 4 activo con `try/except` para no interrumpir la ejecución).
4. **src/graficos.py (Cumplido):** Concentra 29 funciones de visualización (gráficos 00 al 16, tablas y `run()`); los tres notebooks importan de forma centralizada vía `from src import graficos as gr`.
5. **docs/ (Cumplido):** Redactados los tres .md por notebook (`0_exploracion_inicial.md`, `1_eda.md`, `2_modelo.md`), el informe técnico (`informe_tecnico.md`) y el documento técnico en PDF (`Explicacion_Variables_Spotify.pdf`).

## 6. Resultados Clave de 2_Modelacion (corrida final, semilla 42)
- **Modelo oficial (test):** AUC 0.857, gap 0.026, recall 0.833, precisión 0.425, lift@10% 3.36. Cumple los 4 KPIs.
- **Género:** sin género el AUC cae de 0.856 a 0.695; por permutación el género pesa 0.30 de AUC (la siguiente variable, 0.009). El recall por género es muy desigual (0.15 en géneros de hit rate bajo, 0.99 en muy alto); el 51% de los géneros no llega a 0.75.
- **K-Means (audio):** k=2 por silhouette (0.240) separa solo "tranquilo" de "enérgico". Con k=7 (estable, ARI >= 0.957 entre mitades) los clusters son familias de estilo con hit rate entre ≈ 0.10 y ≈ 0.24; usar solo el cluster da AUC 0.576. Los géneros agrupados por audio (k=3) tienen acuerdo moderado entre K-Means y Ward (ARI 0.47); un grupo de 12 géneros acústicos/instrumentales tiene hit rate medio 0.12 frente a 0.245 de los otros.
- **Umbral calibrado en train (sección 10):** máx. F1 en 0.56 (recall 0.691, pierde el KPI); recall 0.78 en train -> umbral ≈ 0.522 (precisión 0.456, recall 0.793, mantiene el KPI).
- **Política suavizada por género (λ=0.1, elegida sin usar test):** recall global 0.795, precisión 0.373, recall en géneros de hit rate < 10% de 0.148 a 0.487, recall macro 0.580 a 0.641. El bootstrap muestra que las diferencias superan el ruido. Umbral por género puro: recall 0.773 (IC 0.759 a 0.785), precisión 0.357.
- **Hiperparámetros (sección 11):** la configuración oficial quedó 9ª de 11 (AUC CV 0.8557). La mejor llegó a 0.8647 en CV y 0.8646 en test (gap 0.039, lift 3.42). La configuración ganadora varía entre entornos; lo estable es la meseta (las cuatro mejores entre 0.8630 y 0.8647, todas con min_samples_leaf = 20).
- **Número de géneros (sección 12):** solo con canciones de un género el AUC es 0.863 (oficial en las mismas canciones, 0.865). El modelo marca más canciones multigénero por composición de géneros.
- **Variables de texto y ensamble (sección 13):** variables de texto +0.003 de AUC en CV (test 0.8594); el ensamble con Regresión Logística empeora (test 0.851).
- **Negocio (sección 14):** el 10% superior del ranking tiene ≈ 69% de hits y captura ≈ 34% de todos los hits; el umbral 0.5 promociona el 40% del catálogo y el 57% de lo promocionado no es hit.

## 7. Decisión Final y Alternativas Documentadas
- Se mantiene el modelo oficial. Quedan documentadas como alternativas, sin adoptar: política suavizada por género, hiperparámetros ajustados, variables de texto y umbral por recall 0.78.
- Ninguna alternativa corrige el sesgo de género sin un costo en algún KPI o sin cambiar el problema. El sesgo queda medido y declarado, no evitado.
- Uso recomendado: priorizar canciones en campañas (tramo superior del ranking). El modelo captura sobre todo qué géneros son populares, no qué hace "buena" a una canción.
- Multi-hot normalizado: evaluado en una corrida previa y descartado (no cambiaba lo suficiente); no tiene celda en el notebook.

## 8. Rúbrica EV2 y Defensa
- Evaluación grupal con defensa individual: 10 minutos de presentación y preguntas cruzadas (cada integrante debe dominar todo el proyecto).
- Entrega: informe técnico .md reproducible tipo README (problema, objetivos, KPIs, fuentes de datos, EDA, CRISP-DM), notebook .ipynb ejecutable, datos y archivos complementarios, carpeta con estructura profesional.
- Requisitos: mínimo 2 modelos supervisados y 1 técnica no supervisada; evaluar el impacto ético.
- Ponderación: IE5 metodología 20%, IE6 modelos supervisados (algoritmo y configuración justificados) 30%, IE7 no supervisado (patrones relevantes y aplicación al negocio) 30%, IE8 interpretación de métricas y recomendaciones de negocio 20%.
- Preguntas probables: por qué no se quitó el género, por qué umbral 0.5, por qué HGB y esa configuración, qué significa silhouette 0.24 y por qué se segmenta con k=7, qué mide el lift, por qué split por canción, qué mide el gap.

## 9. Pendientes Antes de Entregar
1. informe_tecnico.md: confirmar secciones explícitas de Objetivos del proyecto, Impacto ético y mapeo de las fases CRISP-DM (la pauta las exige).
2. Confirmar que la fuente de datos citada en el informe (HuggingFace) coincide con la usada.
3. docs: el multi-hot normalizado figura como probado en 1_eda.md y 2_modelo.md; indicar "evaluado en una corrida previa" o agregar la celda.
4. README.md: objetivo, estructura, ejecución y limitaciones; agregar los gráficos 17 a 19 a los comandos de limpieza.
5. Reproducibilidad final: borrar data/processed, data/raw/train|test y output/modelos; ejecutar python main.py desde cero y "Restart & Run All" en los tres notebooks (la sección 13 de 2_Modelacion lee data/raw/train|test; la búsqueda de hiperparámetros tarda varios minutos).
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
