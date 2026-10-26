# EV2 ML – Spotify Tracks (Caso C) · Notas de contexto para agentes

Fuente de verdad: notebooks y `output/modelos/resultados/*.csv`. Si una cifra de aquí difiere, manda la corrida.

## 1. Problema y KPIs
- Predecir hit (popularidad ≥ P80 de train = 52) para priorizar campañas. Clasificación binaria.
- KPIs (criterios definidos por el equipo/agente, no estándar externo; se mantienen con justificación): AUC ≥ 0.80 · gap AUC train-test < 0.05 · recall ≥ 0.75 · lift@10% ≥ 2.0.
- Debilidades declaradas: no hay meta de precisión; con hit rate ≈ 20% el lift máximo es ≈ 5.

## 2. Datos y preprocesamiento (`src/pipeline.py`)
- 114.000 filas, 114 géneros × 1.000 filas (muestreo no natural). 1 fila con nulos de texto → se elimina.
- Deduplicación agrupada por `track_id`: popularidad máxima, todos los géneros → 89.740 canciones; 18,2% multigénero.
- Multi-hot (`MultiLabelBinarizer`, 114 `genre_*`). `n_generos` excluida (r = 0,104 con hit; posible efecto de la popularidad), pero va implícita en el multi-hot.
- Split después de deduplicar y antes de transformar: `GroupShuffleSplit` por `song_key` (artista+título), seed 42 → 71.706 / 18.034. P80 solo con train.
- `log1p` en speechiness, instrumentalness, liveness, duration_ms (solo duration_ms mejora claramente). Scaler y OHE dentro del `Pipeline`.
- 5 variables de texto sin identidad del artista: n_artistas, t_palabras, t_remix, t_live, t_feat. Total 133 variables.
- Popularidad 0 (10,3% de train; 14,05% de filas crudas) se mantiene como no hit.

## 3. Estructura
- `data/raw|processed/{train,test}` · `notebooks/0_exploracion_inicial, 1_EDA, 2_Modelacion` · `src/pipeline, train, test, metricas (incl. política), graficos` · `main.py` (pipeline → train → test → gráficos).
- `output/graficos/` (00–20, 05b) · `output/modelos/` solo .joblib · `output/modelos/resultados/` todos los CSV (`RESULTADOS_DIR` en `src/train.py`, lo importan test.py y el notebook 2).
- Cachés del notebook 2 (no los genera main.py): `busqueda_hiperparametros.csv` (grilla ≈ 25 min), `comparacion_class_weight.csv`.
- `docs/`: 0_exploracion_inicial.md, 1_eda.md, 2_modelo.md, informe_tecnico.md. La justificación de cada variable está en la sección del EDA donde se muestra su evidencia ("**Decisión:**" en `1_EDA` y `1_eda.md`) (el PDF `Explicacion_Variables_Spotify.pdf` queda fuera de la entrega).

## 4. Modelo oficial
HGB (max_iter 300 como tope: early_stopping="auto" activo por >10.000 filas, 10% interno no agrupado; lr 0.05, profundidad libre, 31 hojas, min_samples_leaf 20, L2 1, class_weight balanced) + política de umbral por género (recall objetivo 0.78, encogimiento k = 30) + corrección por n° de géneros (1, 2, 3+). Calibrada con probabilidades OOF de train. Selección por AUC de CV agrupada (5 folds).

## 5. Resultados oficiales (corrida limpia main.py + 2_Modelacion, 25-10, PC de presentación)
- CV: HGB 0.8669 (gap 0.0338) · RL 0.8414 · RF 0.8238. Ablación: 0.867 con género → 0.706 sin género. Permutación: género 0.31, siguiente 0.009.
- Test oficial: AUC 0.866, gap 0.030, recall 0.795, precisión 0.421, lift 3.43 → cumple los 4. HGB con 0.5: recall 0.824, precisión 0.458. RF no cumple recall (0.686).
- Costo de la política: −4 pts de precisión vs 0.5; −6 pts vs umbral único con igual recall (0.536 → precisión 0.482, recall 0.788).
- Equidad (0.5 → oficial): nicho 0.144 → 0.357 · macro 0.559 → 0.595 · géneros con recall ≥ 0.75 45,9% → 51,4% (49% sigue bajo: 53 de 109) · corr. hit rate–recall 0.95 → 0.84 · recall alto 0.944 → 0.834, muy alto 0.996 → 0.886.
- Por n° de géneros (1 · 2 · 3+): marcadas/hit rate 1.82·1.59·2.18 → 2.04·1.46·1.74; precisión 0.444·0.531·0.416 → 0.391·0.544·0.455; recall 0.797·0.792·0.789. AUC 0.875 · 0.824 · 0.795.
- Bootstrap 300: nicho +0.21 [0.15; 0.26], macro +0.04 [0.02; 0.05], recall −0.03 [−0.04; −0.02], precisión −0.04 [−0.04; −0.03].
- Campaña top 10%: 70% hits, 34% de los hits, solo 1,5% de hits de nicho (68 géneros). Por margen: 14,1%, 91 géneros, lift 2.40.
- class_weight (2b): no mejora el ranking (0.8668 vs 0.8667); a igual recall misma precisión; solo desplaza el umbral.
- Sensibilidades: split por artista −0.008 AUC (cota inferior: 36% del test sigue con artistas vistos) · sin popularidad 0: aporte del género +0.155 → +0.161 (escenarios no comparables) · target relativo al género: AUC 0.67 con género, 0.60 solo audio (lo que el audio predice dentro de un género) · solo 1 género: AUC 0.875.
- Grilla: oficial 0.8661 vs mejor 0.8663 con gap < 0.04 (empate); anterior 25ª de 33. Texto: CV 0.8665 → 0.8634 sin él (+0.003). Ensamble con RL empeora (CV −0.0075, test 0.858).
- K-Means (audio): silhouette máx k=2 (0.240); k=7 estable (ARI ≥ 0.957, cae a 0.677 en k=8), familias de estilo, hit rate 0.10–0.24, AUC solo cluster 0.576. Géneros por audio k=3 (K-Means vs Ward ARI 0.47).
- Alternativas no adoptadas: equidad fuerte (nicho 0.72, precisión 0.276), ranking por margen, fila por (canción, género) 0.849 y reponderar 0.686 (ambas fuera del notebook), multi-hot normalizado (corrida previa, sin celda).

## 6. Criterios de redacción (acordados 25-10)
- EDA cronológico: plantea hipótesis y remite a la modelación sin citar sus resultados. El informe integra ("el EDA sospechó X; la modelación mostró Y").
- No afirmar "antes de mirar test" para 0.78, 30 ni la regla gap < 0.04: fueron decisiones del agente, sin análisis de sensibilidad. Sí es verificable que la calibración usa solo OOF de train.
- La política no "corrige": iguala el recall entre grupos y reduce el sesgo de nicho; 1 género empeora en sobre-marcado. Sesgo en 4 capas: dependencia del género (no se toca), error desigual (reducido), multigénero (redistribuido), campaña top 10% (sigue sesgada).
- Ética: el género no es atributo protegido; se protege la diversidad del catálogo. Evitar "equidad/discriminación" en sentido legal.
- No citar cifras de un sistema operativo concreto: lo que depende de CV puede moverse en el tercer decimal.

## 7. Rúbrica y defensa
- Grupal con defensa individual (10 min + preguntas cruzadas). Entrega: informe .md tipo README, notebooks ejecutables, datos, estructura profesional. Mínimo 2 supervisados + 1 no supervisado + impacto ético.
- Ponderación: IE5 metodología 20% · IE6 supervisados 30% · IE7 no supervisado 30% · IE8 métricas y negocio 20%.
- Preguntas probables: por qué no se quitó el género (sin él AUC 0.706 < KPI; el audio puede actuar de sustituto, no probado) · umbral distinto por género y si es justo · cómo se calibró sin test (OOF; parámetros por criterio) · por qué HGB y esa configuración · silhouette 0.24 y k=7 · lift · split por canción · gap · por qué esos KPIs y por qué no hay meta de precisión · costo de 6 pts de precisión · por qué 300 iteraciones (tope; decide la detención temprana).

## 8. Pendientes
1. Ensayar la defensa (sección 7).

## 9. Hitos (resumen)
- 14-10: modelo mejorado (HGB por CV, texto, política de umbral). 15-10: política movida a `src/metricas.py`; gráfico 20.
- 22/23-10: matriz de correlaciones dividida (05 y 05b); textos del EDA sobre asimetría y `n_generos`.
- 25-10: sección 2b (class_weight); Spearman en EDA; textos de notebooks, docs y README alineados con los criterios de la sección 6; KPIs justificados; CSV movidos a `output/modelos/resultados/`; comentarios de `metricas.py` y `pipeline.py` corregidos; corrida limpia en el PC de presentación da recall 0.795 (las cifras previas 0.791 eran de otra corrida); todo alineado a esta versión. Detención temprana documentada en 2_modelo.md.
