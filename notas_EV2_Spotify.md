# Notas de Trabajo — EV2 Machine Learning (MLY1101)
## Caso C: Spotify — Modelamiento y Evaluación

> **Propósito:** Documento temporal de trabajo para registrar decisiones técnicas, hallazgos y pendientes durante el desarrollo de EV2. **No es el informe técnico final.**
> Última actualización: 2026-09-28 (versión corregida del notebook)

---

## Estado actual del proyecto

| Etapa | Estado | Notas |
|---|---|---|
| Dataset limpio (desde EV1) | ✅ Listo | `data/processed/spotify_clean.csv` — 113.549 filas, 25 col |
| Corrección de preprocesamiento | ✅ Rehecha | Ver sección 1 (v1 solo cambiaba el escalado; v2 corrige duplicados, fuga y codificación) |
| Definición variable objetivo | ✅ Ajustada | hit/no-hit con P80 calculado **solo en train** (52 pts) |
| Notebook `EV2_Spotify_Modelos_corregido.ipynb` | ✅ Ejecutado | Reemplaza a la versión anterior |
| Modelos entrenados (LR, RF, HistGB) | ✅ Listo | Ver sección 4 |
| Evaluación y métricas | ✅ Listo | Ver sección 4 |
| Informe técnico EV2 | 🔜 Pendiente | Usar los resultados de la sección 4 (versión corregida) |
| Presentación PPT EV2 | 🔜 Pendiente | |

---

## 0. Observación del docente (EP1) y qué se hizo

El docente indicó que en EV1 provocamos que una canción con más categorías (géneros) pesara más en el score. La causa: en EV1 se conservaron las filas repetidas por `track_id` (una por género), así que una canción en varios géneros aparecía varias veces en el entrenamiento.

**La versión anterior de EV2 no corregía esto.** Solo dejaba de escalar `track_genre_encoded`, que es otro problema (ver 1.3). La versión corregida sí lo resuelve (ver 1.1).

---

## 1. Decisiones de preprocesamiento

### 1.1 Duplicados por track_id → una fila por canción

| Dato | Valor |
|---|---|
| Filas | 113.549 |
| Canciones únicas (`track_id`) | 89.740 |
| Filas "extra" por repetición | 23.809 (20,97% de las filas) |
| Canciones con 1 género | 73.441 (81,84% de canciones → 64,68% de las filas) |
| Canciones con 2 o más géneros | 18,16% de canciones → **35,32% de las filas** |
| `track_id` con `popularity` distinto entre filas | 720 (0,8%); diferencia mediana 1 pt, máxima 44 pts |
| Filas que comparten `song_key` con otro `track_id` | 13.281 (versiones en otros álbumes) |

**Solución:** una fila por `track_id`; el género pasa a **114 columnas multi-hot** (`genre_<nombre>`). Cuando el `popularity` difiere entre géneros, se usa el promedio redondeado. Solo `popularity` difiere entre filas repetidas; el resto de las columnas es idéntico salvo el género.

`song_key` = artista + nombre de pista en minúsculas; se usa como grupo al separar train/test y en la CV.

### 1.2 Fuga de datos (data leakage) en el split

El `train_test_split` por fila dejaba la misma canción en train y en test. Cuantificado con el mismo Random Forest (100 árboles), mismo umbral 54, Label Encoding del género, cambiando solo el split:

| Split | AUC test |
|---|---|
| A) Por fila (versión anterior) | 0.7958 |
| B) Por canción | 0.7505 |
| **Inflación por fuga** | **+0.0453** |

En el split por fila, el **30,8%** de las filas de test tenían su canción en train. La validación cruzada tenía el mismo problema.

**Solución:** `GroupShuffleSplit` (80/20) y `StratifiedGroupKFold` (5 folds), ambos con `groups=song_key`. Se verifica con un `assert` que ninguna canción aparece en ambos conjuntos.

### 1.3 Codificación y escalado

- La v1 dejó de escalar `track_genre_encoded`, pero **no escalar no elimina el orden falso**: StandardScaler es una transformación lineal y los códigos 0-113 siguen ordenados alfabéticamente. Para árboles escalar nunca importó; para regresión logística el problema seguía intacto.
- La correlación entre el código del género y la popularidad media por género es r = 0.0641 (verificado sobre el dataset), o sea, el orden alfabético no refleja ningún patrón.
- **Solución v2:** one-hot para `key` y `time_signature`, multi-hot para género, `explicit` y `mode` sin transformar.

| Grupo | Variables | Tratamiento |
|---|---|---|
| Continuas (10) | `danceability, energy, loudness, valence, tempo, acousticness, speechiness_log, instrumentalness_log, liveness_log, duration_ms_log` | `StandardScaler` dentro del Pipeline (se ajusta solo con el fold de entrenamiento) |
| Nominales (2) | `key`, `time_signature` | `OneHotEncoder` |
| Binarias (2) | `explicit`, `mode` | Sin transformar |
| Género (114) | `genre_*` multi-hot | Sin transformar |

Total: 128 features antes de codificar. `explicit` mantiene señal real: promedio 36.52 vs 33.02 pts (+3.5, verificado).

### 1.4 Otros ajustes
- **Umbral P80 solo con train** (antes se calculaba con todo el dataset): **52 pts** (antes 54).
- **StandardScaler dentro de un `Pipeline`**: se ajusta con train en el split y con cada fold de train en la CV (en EV1 se ajustaba sobre todo el dataset).

### Datos de referencia (EV1)
| Dato | Valor |
|---|---|
| Dataset crudo | 114.000 filas |
| Dataset limpio | 113.549 filas (-451: 1 nulo + 450 duplicados exactos) |
| Variación en medias post-limpieza | < 0.1% en todas las métricas |
| Géneros únicos | 114 (exactamente 1.000 filas por género en el raw) |

---

## 2. Estrategia de modelamiento

### Cambio de estrategia: Regresión → Clasificación binaria

**Motivo del cambio:**
- Las features acústicas tienen correlación lineal muy débil con popularidad (ninguna supera r=0.06)
- R² esperado en regresión sería < 0.10
- Lo que el sello discográfico necesita saber: **¿invierto o no en esta canción?** → decisión binaria

**Variable objetivo:**
```python
P80 = train['popularity'].quantile(0.80)   # 52.0, calculado solo con train
hit = 1 si popularity >= P80 else 0
```

- Train: 71.706 canciones, hits 21,0%
- Test: 18.034 canciones, hits 20,5%

> **Balance de clases (~80/20):** `class_weight='balanced'` en los tres modelos. **Corrección a lo anotado antes:** GBM (`GradientBoostingClassifier`) **no** maneja el desbalance internamente; por eso tenía recall 0.16. Se reemplazó por `HistGradientBoostingClassifier(class_weight='balanced')`.

### Modelos a comparar

| # | Modelo | Rol | Config clave |
|---|---|---|---|
| 1 | **Regresión Logística** | Baseline interpretable | `max_iter=2000`, `class_weight='balanced'` |
| 2 | **Random Forest** | Ensamble principal | `n_estimators=200`, `max_depth=15`, `min_samples_leaf=10`, `class_weight='balanced'` |
| 3 | **HistGradientBoosting** | Ensamble avanzado | `max_iter=300`, `lr=0.05`, `max_depth=6`, `class_weight='balanced'` |

### Métricas de evaluación
- **Principal:** AUC-ROC (insensible al umbral de clasificación)
- **Negocio:** F1-Score en clase `hit=1`
- **Validación cruzada:** `StratifiedGroupKFold` 5-fold sobre train, agrupando por canción
- **Criterio de sobreajuste:** gap AUC train-test < 0.05

### Split
- 80% entrenamiento / 20% prueba **por canción** (`GroupShuffleSplit`, `random_state=42`)
- No es estratificado (no se puede estratificar por grupos con `GroupShuffleSplit`); las proporciones de hits quedaron 21,0% / 20,5%

---

## 3. Archivos generados

| Archivo | Descripción |
|---|---|
| `notebooks/EV2_Spotify_Modelos_corregido.ipynb` | Notebook principal (versión corregida, reemplaza a `EV2_Spotify_Modelos.ipynb`) |
| `images/peso_canciones_multigenero.png` | % de canciones vs % de filas según nº de géneros |
| `images/comparativa_modelos_roc_corregido.png` | Curvas ROC + barras de métricas |
| `images/matrices_confusion_corregido.png` | Matrices de confusión de los 3 modelos |
| `images/feature_importance_rf_corregido.png` | Importancia de variables agrupada (Random Forest) |

El notebook corregido ya no guarda `X_train_corrected.csv` ni los demás CSV de la versión anterior: el preprocesamiento vive dentro del `Pipeline`.

---

## 4. Resultados de los modelos (versión corregida)

> Ejecutado sobre el conjunto de prueba de **18.034 canciones no vistas**, con las 5 correcciones de la sección 1. Falta re-ejecutarlo en la máquina local para confirmar que los números coinciden.

| Modelo | Accuracy | Precision | Recall | F1-Score | AUC-ROC | AUC train |
|---|---|---|---|---|---|---|
| Regresión Logística | 0.7338 | 0.4216 | 0.7979 | 0.5517 | 0.8357 | 0.8404 |
| Random Forest | 0.7283 | 0.4126 | 0.7631 | 0.5356 | 0.8241 | 0.8403 |
| HistGradientBoosting | 0.7327 | 0.4228 | 0.8271 | **0.5595** | **0.8544** | 0.8812 |

**Validación cruzada 5-fold por canción (sobre train):**

| Modelo | AUC test (CV) | AUC train (CV) | Gap AUC | F1 (CV) | Precision (CV) | Recall (CV) |
|---|---|---|---|---|---|---|
| Regresión Logística | 0.8377 ± 0.0039 | 0.8407 | 0.0030 | 0.5637 ± 0.0065 | 0.4342 | 0.8033 |
| Random Forest | 0.8207 ± 0.0039 | 0.8421 | 0.0214 | 0.5366 ± 0.0056 | 0.4236 | 0.7334 |
| HistGradientBoosting | 0.8563 ± 0.0025 | 0.8850 | 0.0287 | 0.5695 ± 0.0051 | 0.4349 | 0.8252 |

Los tres modelos quedan bajo el umbral de 0.05: sin sobreajuste significativo. El gap del Random Forest bajó de 0.1059 a 0.0214.

**Comparación con la versión anterior** (indicativa: cambian el umbral, las filas y el conjunto de test):

| Modelo | AUC anterior | AUC corregido | F1 anterior | F1 corregido | Recall anterior | Recall corregido |
|---|---|---|---|---|---|---|
| Regresión Logística | 0.6257 | 0.8357 | 0.3757 | 0.5517 | 0.6455 | 0.7979 |
| Random Forest | 0.7981 | 0.8241 | 0.5117 | 0.5356 | 0.7384 | 0.7631 |
| Gradient Boosting | 0.8271 | 0.8544 | 0.2561 | 0.5595 | 0.1538 | 0.8271 |

**Análisis:**

| Modelo | Fortaleza | Debilidad |
|---|---|---|
| Regresión Logística | Gap 0.003; su salto de AUC (0.6257 → 0.8357) confirma que el problema era el Label Encoding del género | Precision 0.42 (muchos falsos positivos) |
| Random Forest | Estable, gap 0.0214 | Menor AUC y F1 de los tres |
| HistGradientBoosting | **Mejor AUC, F1 y recall** | Mayor gap (0.0287), aunque bajo el criterio |

> **Modelo recomendado: HistGradientBoosting** — mejor AUC (métrica principal declarada), mejor F1 y recall 0.83. Esto cambia la recomendación anterior (Random Forest), que se basaba en un GBM sin balanceo de clases y en métricas con fuga.

**Importancia de variables (Random Forest, agrupada):**

| Variable | Importancia |
|---|---|
| Género (multi-hot, suma de 114 columnas) | 0.7337 |
| `instrumentalness_log` | 0.0451 |
| `duration_ms_log` | 0.0421 |
| `liveness_log` | 0.0420 |
| `energy` | 0.0262 |

> **Ojo en la defensa:** el género concentra el 73,4% de la importancia y la regresión logística queda cerca de HistGB (0.8357 vs 0.8544). Gran parte del poder predictivo viene del género y no de las características de audio.

---

## 5. Pendientes / próximos pasos

- [x] Completar la sección 4 con resultados reales del notebook
- [x] Revisar feature importance: el género domina (0.7337)
- [x] Reducir sobreajuste en Random Forest: gap 0.106 → 0.0214 (con el split por canción y el nuevo preprocesamiento)
- [x] Mejorar recall de Gradient Boosting: resuelto con `class_weight='balanced'` (recall 0.15 → 0.83)
- [x] Evaluar alternativa a Label Encoding: se usó multi-hot/one-hot (Target Encoding ya no hace falta)
- [ ] Re-ejecutar `EV2_Spotify_Modelos_corregido.ipynb` en la máquina local y confirmar que los números coinciden
- [ ] Decidir si hacer tuning de hiperparámetros (GridSearchCV / RandomizedSearchCV)
- [ ] Analizar errores del modelo (¿qué géneros falla más en predecir?)
- [ ] Medir el aporte real de las features de audio: entrenar un modelo sin las columnas de género y comparar
- [ ] Decidir el umbral de decisión según costo de negocio (hoy 0.5)
- [ ] Escribir el Informe Técnico EV2 en formato formal (incluir la sección 0 y la cuantificación de la fuga)
- [ ] Crear presentación PPT EV2
- [ ] Preparar la defensa: explicar qué observó el docente, por qué la v1 de EV2 no lo corregía y cómo se cuantificó la fuga

---

## 6. Preguntas / dudas a resolver

- ¿El profesor espera regresión Y clasificación, o solo clasificación?
- ¿Cuántos modelos mínimos pide la pauta EV2?
- ¿Se requiere tuning de hiperparámetros o basta con configuración razonada?
- ¿La comparativa EV1 (raw vs clean) debe estar en el informe EV2 también o solo en EV1?
- ¿El docente acepta el cambio de `GradientBoostingClassifier` a `HistGradientBoostingClassifier`, o hay que justificarlo en el informe?
