# Notas de Trabajo — EV2 Machine Learning (MLY1101)
## Caso C: Spotify — Modelamiento y Evaluación

> **Propósito:** Documento temporal de trabajo para registrar decisiones técnicas, hallazgos y pendientes durante el desarrollo de EV2. **No es el informe técnico final.**
> Última actualización: 2026-09-30 (versión corregida del notebook + negocio, KPIs y K-Means + ablación audio vs género, umbral por género y Target Encoding: secciones 4.4 y 8)

---

## Estado actual del proyecto

| Etapa | Estado | Notas |
|---|---|---|
| Dataset limpio (desde EV1) | ✅ Listo | `data/processed/spotify_clean.csv` — 113.549 filas, 25 col |
| Corrección de preprocesamiento | ✅ Rehecha | Ver sección 1 (v1 solo cambiaba el escalado; v2 corrige duplicados, fuga y codificación) |
| Definición variable objetivo | ✅ Ajustada | hit/no-hit con P80 calculado **solo en train** (52 pts) |
| Notebook `EV2_Spotify_Modelos.ipynb` | ✅ Ejecutado | Versión corregida + sección 0 (negocio, CRISP-DM, KPIs), justificaciones, KPIs de negocio y K-Means |
| Modelo no supervisado (K-Means, 6 segmentos) | ✅ Listo | Sección 9 del notebook; la pauta lo exige (IE7, 30%) |
| Modelos entrenados (LR, RF, HistGB) | ✅ Listo | Ver sección 4 |
| Evaluación y métricas | ✅ Listo | Ver sección 4 |
| Dominio del género: ablación y alternativas | ✅ Hecho | Sección 4.4 (resultados) y sección 8 (historial de problemas y justificación). Notebook 7.3 a 7.6 |
| Informe técnico `.md` tipo README | 🔜 Pendiente | **Entregable obligatorio de la pauta** (ver sección 7) |
| Estructura de carpetas (data/, notebooks/, models/, images/, README.md) | 🔜 Por revisar | Exigida por la pauta |
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
| `notebooks/EV2_Spotify_Modelos.ipynb` | Notebook principal (versión corregida, con negocio/CRISP-DM, KPIs y no supervisado) |
| `models/*.joblib` | Pipelines serializados (LR, RF, HistGB y `best_model.joblib`) |
| `images/curva_ganancia_acumulada.png` | Curva de ganancia acumulada (KPIs de negocio) |
| `images/kmeans_seleccion_k.png`, `kmeans_segmentos_pca.png`, `kmeans_perfil_segmentos.png` | Selección de K, segmentos en PCA y perfil de centroides |
| `images/peso_canciones_multigenero.png` | % de canciones vs % de filas según nº de géneros |
| `images/comparativa_modelos_roc_corregido.png` | Curvas ROC + barras de métricas |
| `images/matrices_confusion_corregido.png` | Matrices de confusión de los 3 modelos |
| `images/feature_importance_rf_corregido.png` | Importancia de variables agrupada (Random Forest) |

El notebook corregido ya no guarda `X_train_corrected.csv` ni los demás CSV de la versión anterior: el preprocesamiento vive dentro del `Pipeline`.

---

## 4. Resultados de los modelos (versión corregida)

> Ejecutado sobre el conjunto de prueba de **18.034 canciones no vistas**, con las correcciones de la sección 1. Las métricas de test coinciden con la ejecución local; la validación cruzada difiere en el tercer decimal y aquí se usan los números de la ejecución local.

| Modelo | Accuracy | Precision | Recall | F1-Score | AUC-ROC | AUC train |
|---|---|---|---|---|---|---|
| Regresión Logística | 0.7338 | 0.4216 | 0.7979 | 0.5517 | 0.8357 | 0.8404 |
| Random Forest | 0.7283 | 0.4126 | 0.7631 | 0.5356 | 0.8241 | 0.8403 |
| HistGradientBoosting | 0.7327 | 0.4228 | 0.8271 | **0.5595** | **0.8544** | 0.8812 |

**Validación cruzada 5-fold por canción (sobre train):**

| Modelo | AUC test (CV) | AUC train (CV) | Gap AUC | F1 (CV) | Precision (CV) | Recall (CV) |
|---|---|---|---|---|---|---|
| Regresión Logística | 0.8380 ± 0.0053 | 0.8406 | 0.0026 | 0.5640 ± 0.0083 | 0.4342 | 0.8045 |
| Random Forest | 0.8224 ± 0.0055 | 0.8422 | 0.0198 | 0.5373 ± 0.0097 | 0.4200 | 0.7463 |
| HistGradientBoosting | 0.8566 ± 0.0034 | 0.8847 | 0.0280 | 0.5677 ± 0.0079 | 0.4327 | 0.8253 |

Los tres modelos quedan bajo el umbral de 0.05: sin sobreajuste significativo. El gap del Random Forest bajó de 0.1059 a 0.0198.

**Comparación con la versión anterior** (indicativa: cambian el umbral, las filas y el conjunto de test):

| Modelo | AUC anterior | AUC corregido | F1 anterior | F1 corregido | Recall anterior | Recall corregido |
|---|---|---|---|---|---|---|
| Regresión Logística | 0.6257 | 0.8357 | 0.3757 | 0.5517 | 0.6455 | 0.7979 |
| Random Forest | 0.7981 | 0.8241 | 0.5117 | 0.5356 | 0.7384 | 0.7631 |
| Gradient Boosting | 0.8271 | 0.8544 | 0.2561 | 0.5595 | 0.1538 | 0.8271 |

**Análisis:**

| Modelo | Fortaleza | Debilidad |
|---|---|---|
| Regresión Logística | Gap 0.0026; su salto de AUC (0.6257 → 0.8357) confirma que el problema era el Label Encoding del género | Precision 0.42 (muchos falsos positivos) |
| Random Forest | Estable, gap 0.0198 | Menor AUC y F1 de los tres |
| HistGradientBoosting | **Mejor AUC, F1 y recall** | Mayor gap (0.0280), aunque bajo el criterio |

> **Modelo recomendado: HistGradientBoosting** — mejor AUC (métrica principal declarada), mejor F1 y recall 0.83. Esto cambia la recomendación anterior (Random Forest), que se basaba en un GBM sin balanceo de clases y en métricas con fuga.

**Importancia de variables (Random Forest, agrupada):**

| Variable | Importancia |
|---|---|
| Género (multi-hot, suma de 114 columnas) | 0.7337 |
| `instrumentalness_log` | 0.0451 |
| `duration_ms_log` | 0.0421 |
| `liveness_log` | 0.0420 |
| `energy` | 0.0262 |

> **Ojo en la defensa:** el género concentra el 73,4% de la importancia y la regresión logística queda cerca de HistGB (0.8357 vs 0.8544). Gran parte del poder predictivo viene del género y no de las características de audio. **Esto se cuantificó con una ablación (sección 4.4):** sin género el AUC cae a 0.63–0.70.

### 4.1 KPIs de negocio (sección 7.1 del notebook)

Metas de referencia definidas por el equipo: AUC-ROC ≥ 0.80, Recall ≥ 0.75, Lift @ top-10% ≥ 2.0. Los tres modelos las cumplen. Tasa base de hits en test: 20.53%.

| Modelo | Lift umbral 0.5 | Precision top-10% | Lift @ top-10% | % hits capturados top-10% | % hits capturados top-20% |
|---|---|---|---|---|---|
| Regresión Logística | 2.05 | 0.6184 | 3.01 | 30.1% | 52.5% |
| Random Forest | 2.01 | 0.6484 | 3.16 | 31.6% | 52.9% |
| HistGradientBoosting | 2.06 | 0.6772 | **3.30** | **33.0%** | **56.2%** |

**Lectura para el negocio:** con umbral 0.5, solo el 42% de las canciones marcadas son hits (más de la mitad son falsos positivos). Sirve para **priorizar**: revisando el 10% mejor puntuado se encuentran 3.3 veces más hits que al azar. No es una autorización automática de inversión.

### 4.2 Segmentación no supervisada (K-Means, sección 9 del notebook)

- Variables: las 10 audio features estandarizadas (sin género ni popularidad, para que no sea circular); ajustado solo con train; test se asigna con `predict`.
- K = 6: silueta 0.159, estabilidad ARI = 0.997. El codo no tiene quiebre nítido y la silueta es máxima en k = 2 (0.240), pero dos grupos no sirven. K = 6 es un compromiso entre interpretabilidad y estabilidad, no un óptimo estadístico. Silueta baja = fronteras difusas.

| Segmento | Hit rate test | Lift vs base | Géneros frecuentes |
|---|---|---|---|
| Bailable y positivo | 23.8% | 1.16x | latino, reggaeton, salsa |
| Acústico suave | 22.7% | 1.11x | tango, honky-tonk, jazz |
| Rock/metal de alta energía | 21.1% | 1.03x | metalcore, heavy-metal, grunge |
| Hablado / alta speechiness | 17.8% | 0.87x | comedy, j-dance, dancehall |
| Instrumental de ambiente | 17.4% | 0.85x | sleep, new-age, classical |
| Electrónica instrumental | 10.2% | 0.50x | minimal-techno, detroit-techno, techno |

**Lectura:** las diferencias son moderadas (10.2% a 23.8%) y solo Electrónica instrumental se aleja claramente de la tasa base. Los segmentos se solapan con familias de género. Que un segmento tenga menos hits en Spotify no implica menos valor para el negocio (hipótesis no evaluada).

### 4.3 Diagnóstico: número de géneros vs puntaje del modelo (sección 7.2 del notebook)

Hay dos mecanismos posibles detrás de la observación del docente: (1) el **peso en el entrenamiento** (filas repetidas) y (2) el **puntaje del modelo**.

- **(1) Corregido:** 0 filas repetidas por `track_id` en train; cada canción pesa una vez.
- **(2) El puntaje sí sube con el número de géneros** (modelo principal, test):

| Géneros de la canción | Canciones | Hit rate real | Popularity media | Probabilidad media |
|---|---|---|---|---|
| 1 | 14.678 | 17,8% | 32,1 | 0,381 |
| 2 | 2.286 | 34,6% | 39,7 | 0,517 |
| 3 o más | 1.070 | 27,9% | 28,3 | 0,543 |

Correlación n° de géneros vs probabilidad: 0,205; vs popularity real: −0,004.

**Prueba de si el modelo "cuenta géneros"** (AUC / correlación n° géneros–probabilidad):

| Representación del género | Regresión Logística | Gradient Boosting |
|---|---|---|
| A) Multi-hot (actual) | 0,8357 / 0,209 | 0,8544 / 0,205 |
| B) Proporciones (suma 1) | 0,8376 / 0,215 | 0,8594 / 0,217 |
| C) Un solo género por canción | 0,8324 / 0,236 | 0,8437 / 0,219 |

**Conclusión:** neutralizar el conteo **no elimina** la relación (con un solo género por canción el modelo no puede contar y la correlación sigue en 0,22–0,24). El mayor puntaje viene de que las canciones multi-género son distintas y más exitosas en este dataset (34,6% de hits con 2 géneros vs 17,8% con 1), no de que el modelo cuente categorías. Se mantiene el multi-hot (A). Las diferencias de AUC (−0,011 a +0,005) son pequeñas frente a la desviación de la CV (0,003–0,005).

**Para la defensa:** el fix del docente fue la deduplicación (una fila por canción). El grupo de 3 o más géneros recibe la probabilidad más alta aunque su tasa real es menor que la de 2 géneros; es una sobreestimación moderada en un grupo pequeño. Hipótesis no evaluada: canciones en más listas son más populares, o el muestreo por género (~1.000 canciones por género) favorece a las multi-género populares.

---

### 4.4 ¿Cuánto aporta el audio frente al género? Ablación y alternativas (secciones 7.3 a 7.6 del notebook)

**Por qué se hizo.** La importancia de variables (73.4% para el género) no dice cuánto rinde el modelo *sin* esas variables. Un agente de apoyo propuso tres caminos para "resolver el sesgo" del género; en lugar de aceptarlos sin verificar, se evaluó cada uno con los mismos datos, split por canción (`GroupShuffleSplit`, `random_state=42`), Pipeline, hiperparámetros y umbral P80 = 52 (solo train).

**(a) Ablación: modelo sin columnas de género (HECHO, es el argumento principal).**

| Modelo | Variante | AUC | F1 | Recall | Lift @ top-10% |
|---|---|---|---|---|---|
| Regresión Logística | A) Todo | 0.8357 | 0.5517 | 0.7979 | 3.01 |
| Regresión Logística | B) Solo audio | 0.6324 | 0.3725 | 0.6175 | 1.62 |
| Regresión Logística | C) Solo género | 0.8314 | 0.5433 | 0.7861 | 3.00 |
| HistGradientBoosting | A) Todo | 0.8544 | 0.5595 | 0.8271 | 3.30 |
| HistGradientBoosting | B) Solo audio | 0.6952 | 0.4192 | 0.6831 | 2.05 |
| HistGradientBoosting | C) Solo género | 0.8393 | 0.5648 | 0.7326 | 3.20 |

- Sin género el AUC cae **0.2033 (LR)** y **0.1592 (HistGB)**. La variación de la CV es 0.003–0.005, así que la caída no es ruido.
- Saber solo el género casi iguala al modelo completo. El audio suma **+0.004 (LR)** y **+0.015 (HistGB)** de AUC sobre el género.
- El audio sí tiene señal (AUC 0.63–0.70 > 0.50), pero no alcanza las metas: AUC < 0.80; solo HistGB llega justo al Lift ≥ 2.0 (2.05).
- **Matiz que hay que decir en la defensa:** B *subestima* el aporte total del audio, porque el audio también informa el género (K-Means recupera familias de género desde el audio). Y no prueba que el género *cause* la popularidad: puede reflejar exposición algorítmica (sesgo de EP1).

**(b) Umbral de hit relativo al género (HECHO, solo diagnóstico).** P80 por género calculado solo con train, usando el primer género de cada canción.

| Modelo | Variables | AUC | Lift @ top-10% |
|---|---|---|---|
| HistGB, target relativo | Todo | 0.6491 | 1.96 |
| HistGB, target relativo | Solo audio | 0.5730 | 1.41 |

Tasa de hits relativa: 21.7% (train) / 21.6% (test). Ningún género necesitó el `clip(lower=1)`. Dentro de un mismo género el audio casi no separa hits de no-hits (0.573). **No reemplaza el target principal** (ver problemas en sección 8).

**(c) Target Encoding regularizado (HECHO, solo robustez).** Una columna en lugar de 114, suavizado con m = 20, estimación out-of-fold en train (`StratifiedGroupKFold` por canción) y promedio de los géneros en canciones multi-género.

| Modelo | AUC | F1 | Recall | Lift @ top-10% | AUC con multi-hot |
|---|---|---|---|---|---|
| HistGB (audio + TE) | 0.8482 | 0.5633 | 0.7985 | 3.18 | 0.8544 |
| LR (audio + TE) | 0.8335 | 0.5554 | 0.7458 | 3.10 | 0.8357 |

**Decisión:** el modelo principal sigue siendo **HistGradientBoosting con multi-hot y target global (P80 de train)**. (b) y (c) no lo mejoran y quedan como evidencia complementaria. El género dominante se reporta como **hallazgo y limitación**, no como error corregido.

---

## 5. Pendientes / próximos pasos

- [x] Completar la sección 4 con resultados reales del notebook
- [x] Revisar feature importance: el género domina (0.7337)
- [x] Reducir sobreajuste en Random Forest: gap 0.106 → 0.0198 (con el split por canción y el nuevo preprocesamiento)
- [x] Mejorar recall de Gradient Boosting: resuelto con `class_weight='balanced'` (recall 0.15 → 0.83)
- [x] Evaluar alternativa a Label Encoding: se usó multi-hot/one-hot; luego se probó Target Encoding regularizado (sección 4.4c): mismo rendimiento, se mantiene multi-hot
- [x] Re-ejecutar el notebook en la máquina local: las métricas de test coinciden
- [x] Agregar sección de negocio/CRISP-DM/KPIs, justificación de variables y modelos, modelo no supervisado y conclusiones de negocio al notebook
- [x] Medir si el puntaje depende del número de géneros (sección 7.2): no depende de que el modelo cuente géneros
- [ ] Volver a ejecutar en la máquina local la versión nueva del notebook (incluye K-Means y secciones 7.2 a 7.6) y confirmar que los números de las secciones 4.2, 4.3 y 4.4 coinciden
- [ ] Decidir si hacer tuning de hiperparámetros (GridSearchCV / RandomizedSearchCV)
- [ ] Analizar errores del modelo (¿qué géneros falla más en predecir?)
- [x] Medir el aporte real de las features de audio: entrenar un modelo sin las columnas de género y comparar (sección 4.4: AUC cae 0.16–0.20)
- [ ] Decidir el umbral de decisión según costo de negocio (hoy 0.5)
- [ ] Escribir el **informe técnico en Markdown (.md) tipo README**, entregable obligatorio: problema de negocio, objetivos, KPIs, fuentes de datos, EDA, CRISP-DM (ver sección 7)
- [ ] Revisar que la carpeta del proyecto tenga la estructura pedida: `data/`, `notebooks/`, `models/`, `images/`, `README.md`, y que el notebook corra sin modificaciones
- [ ] Confirmar con el docente las metas de los KPIs (las definió el equipo como referencia)
- [ ] Crear la presentación (10 minutos por estudiante) con los 5 apartados pedidos
- [ ] Preparar la defensa: explicar qué observó el docente, por qué la v1 de EV2 no lo corregía, cómo se cuantificó la fuga y cómo se midió el dominio del género (usar la sección 8 como guion)

---

## 6. Preguntas / dudas a resolver

- ¿El profesor espera regresión Y clasificación, o solo clasificación?
- ~~¿Cuántos modelos mínimos pide la pauta EV2?~~ Respondida: mínimo **2 supervisados + 1 no supervisado**.
- ¿Se requiere tuning de hiperparámetros o basta con configuración razonada?
- ¿La comparativa EV1 (raw vs clean) debe estar en el informe EV2 también o solo en EV1?
- ¿El docente acepta el cambio de `GradientBoostingClassifier` a `HistGradientBoostingClassifier`, o hay que justificarlo en el informe?
- ¿El docente considera el dominio del género un problema a corregir o un hallazgo válido a reportar? (hoy se reporta como hallazgo, con ablación)

---

## 7. Pauta de EP2 (40%) y cobertura en el notebook

**Formato:** grupal con **defensa individual** (10 min por estudiante, preguntas cruzadas). 5 horas en sala de proyectos.

**Entregables:** (1) informe técnico `.md` tipo README, (2) notebook ejecutable y documentado, (3) datos y archivos complementarios para re-ejecutar sin modificaciones, (4) carpeta con estructura profesional (`data/`, `notebooks/`, `models/`, `images/`, `README.md`).

**Secciones mínimas del informe `.md`:** problema de negocio, objetivos, KPIs, fuentes de datos, preparación y EDA, metodología CRISP-DM.

**Apartados de la presentación:** problema de negocio/KPIs/objetivos; EDA; preparación y transformación; implementación y comparación de modelos; evaluación con métricas apropiadas.

| Indicador | Peso | Qué pide el nivel máximo | Dónde se cubre | Estado |
|---|---|---|---|---|
| IE5 Metodología | 20% | Aplicar CRISP-DM estructurando coherentemente todas las etapas | Notebook sección 0.5 (mapa de fases) y ejemplo de iteración con la observación del docente | ✅ Cubierto en el notebook; falta en el informe `.md` |
| IE6 Modelos supervisados | 30% | Modelos adecuados, **justificando algoritmo y configuración** | Secciones 5.1 y 5.2, 6 y 7 | ✅ Cubierto (hiperparámetros justificados pero sin búsqueda sistemática) |
| IE7 No supervisado | 30% | Implementar, identificar patrones relevantes y justificar su aplicación al negocio | Sección 9 (K-Means, 6 segmentos, hit rate por segmento) | ✅ Cubierto; fronteras difusas (silueta 0.16) declaradas |
| IE8 Interpretación | 20% | Interpretar métricas y traducirlas en recomendaciones de negocio | Sección 7.1 y sección 12 | ✅ Cubierto |

---

## 8. Historial de problemas encontrados y justificación de cada decisión (guion para la defensa)

Cada fila es un problema real que apareció en una versión anterior, por qué esa primera solución no bastaba y qué se hizo. Las cifras están medidas en este proyecto (no son supuestos).

### 8.1 Problemas de las primeras versiones (v1 de EV2 y EV1)

| # | Problema detectado | Primera solución (insuficiente) | Por qué no bastaba | Solución final | Evidencia |
|---|---|---|---|---|---|
| 1 | Canciones repetidas por género pesaban más (observación del docente) | v1 solo dejó de escalar `track_genre_encoded` | No tocaba las filas repetidas; eran 23.809 filas extra (20.97%) | Una fila por `track_id`, género multi-hot (114 columnas) | 0 filas repetidas en train |
| 2 | Orden falso del género (Label Encoding) | No escalar el código | Escalar es una transformación lineal: el orden alfabético 0–113 sigue ahí y la regresión logística lo interpreta como magnitud (r con popularidad media = 0.0641) | One-hot (`key`, `time_signature`) y multi-hot (género) | LR: AUC 0.6257 → 0.8357 |
| 3 | Fuga de datos en el split | `train_test_split` por fila | 30.8% de las filas de test tenían su canción en train; la CV tenía el mismo problema | `GroupShuffleSplit` y `StratifiedGroupKFold` con `groups=song_key` + `assert` de disjunción | AUC inflado en +0.0453 (0.7958 vs 0.7505) |
| 4 | Umbral P80 calculado con todo el dataset | P80 global = 54 | Usaba información de test para definir el target | P80 solo con train (52 pts) | Train 21.0% / test 20.5% de hits |
| 5 | `StandardScaler` ajustado con todo el dataset (EV1) | Escalar antes del split | Contamina train con estadísticos de test | `StandardScaler` dentro del `Pipeline` (se ajusta por fold) | — |
| 6 | Gradient Boosting con recall 0.16 | `GradientBoostingClassifier` sin ajustes | No maneja el desbalance (~80/20) internamente | `HistGradientBoostingClassifier(class_weight='balanced')` | Recall 0.15 → 0.83 |
| 7 | Random Forest sobreajustado (gap 0.106) | Hiperparámetros por defecto | Árboles profundos memorizaban, además de la fuga | `max_depth=15`, `min_samples_leaf=10` + split por canción | Gap 0.106 → 0.0198 |

### 8.2 El conteo de géneros (sección 4.3 y 7.2)

- **Problema:** el puntaje del modelo sube con el número de géneros (0.381 / 0.517 / 0.543 para 1 / 2 / 3+).
- **Primera hipótesis:** el modelo "cuenta categorías". **Se descartó** con proporciones (B) y un solo género por canción (C): la correlación persiste (0.22–0.24).
- **Explicación respaldada por los datos:** las canciones multi-género son distintas (34.6% de hits con 2 géneros vs 17.8% con 1). Una hipótesis no evaluada: muestreo del dataset (~1.000 canciones por género).

### 8.3 El género domina el modelo: las tres propuestas del agente y qué se encontró

Planteamiento original del agente: "resolver el sesgo" del género. Se evaluó cada propuesta antes de aceptarla:

| Propuesta | Problema o riesgo encontrado al revisarla | Qué se hizo | Resultado | Decisión |
|---|---|---|---|---|
| 1. Entrenar sin género y comparar AUC | El agente dio una cifra ilustrativa ("0.85 → 0.65") sin haberla medido. Además B subestima el aporte del audio (el audio informa el género) | Ablación A/B/C con LR y HistGB, mismo split y pipeline | AUC 0.8544 → 0.6952 (HistGB), 0.8357 → 0.6324 (LR); solo género: 0.8393 / 0.8314 | **Adoptada.** Es el argumento principal |
| 2. P80 por género | (i) Una canción multi-género no tiene un percentil único; (ii) el target cambia de "popular en Spotify" a "popular para su género", así que invalida la comparación con los KPIs; (iii) un género con P80 = 0 haría hit a todas sus canciones | Primer género alfabético (convención declarada), P80 solo con train, `clip(lower=1)` | AUC 0.6491 (todo) y 0.5730 (solo audio) | **Solo diagnóstico.** Mantener el target global |
| 3. Target Encoding regularizado | (i) Usa el target: si no es out-of-fold hay fuga; (ii) no "evita aprender el diccionario de géneros": la columna única *es* ese diccionario comprimido | TE con suavizado m = 20, OOF con `StratifiedGroupKFold`, promedio para multi-género | AUC 0.8482 (HistGB) y 0.8335 (LR), casi igual que multi-hot | **Solo robustez.** Mantener multi-hot |

### 8.4 Por qué se mantiene el modelo final

1. **Target global (P80 de train):** responde la pregunta de negocio original ("¿invierto en esta canción?") y es comparable con los KPIs (AUC ≥ 0.80, Recall ≥ 0.75, Lift ≥ 2.0).
2. **Multi-hot:** no pierde rendimiento frente a alternativas (TE, proporciones, un solo género), conserva el efecto de cada género y no usa el target para construir features.
3. **HistGradientBoosting:** mejor AUC (0.8544), F1 (0.5595) y recall (0.8271); gap 0.0280 < 0.05.
4. **El género dominante se reporta, no se "corrige":** es una propiedad real del dataset (y de la plataforma), cuantificada con la ablación. Recomendación de negocio: usar el modelo para *priorizar* y no para descartar géneros con poca exposición.

### 8.5 Preguntas probables en la defensa

| Pregunta | Respuesta corta |
|---|---|
| ¿Qué observó el docente y qué hicieron? | Canciones con más géneros pesaban más por filas repetidas; se dejó una fila por canción con multi-hot. |
| ¿Por qué la v1 no lo corregía? | Solo dejó de escalar el código de género; no eliminaba filas repetidas ni el orden falso. |
| ¿Cómo saben que había fuga? | Mismo RF, mismo umbral, solo cambia el split: por fila 0.7958, por canción 0.7505 (+0.0453). |
| ¿El modelo predice calidad de audio o popularidad del género? | Sobre todo género: sin género AUC 0.63–0.70; solo género 0.83–0.84. El audio suma +0.004 a +0.015. |
| ¿No es eso un sesgo que debieron corregir? | Se probaron el umbral por género y el Target Encoding; no mejoran ni cambian la conclusión. Se reporta como limitación y recomendación de uso. |
| ¿El audio no sirve para nada? | Sí sirve (AUC 0.63–0.70 > 0.50), pero es insuficiente solo; además el audio informa el género, así que la ablación lo subestima. |
| ¿Por qué no usar el target relativo al género? | Cambia la pregunta de negocio y no es comparable con los KPIs; además requiere elegir un género único por canción. |

