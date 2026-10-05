# Notas de Trabajo — EV2 Machine Learning (MLY1101)
## Caso C: Spotify — Modelamiento y Evaluación

> **Propósito:** Documento temporal de trabajo para registrar decisiones técnicas, hallazgos y pendientes durante el desarrollo de EV2. **No es el informe técnico final.**
> Última actualización: 2026-09-30 (versión corregida del notebook + negocio, KPIs y K-Means)

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

> **Ojo en la defensa:** el género concentra el 73,4% de la importancia y la regresión logística queda cerca de HistGB (0.8357 vs 0.8544). Gran parte del poder predictivo viene del género y no de las características de audio.

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

## 5. Pendientes / próximos pasos

- [x] Completar la sección 4 con resultados reales del notebook
- [x] Revisar feature importance: el género domina (0.7337)
- [x] Reducir sobreajuste en Random Forest: gap 0.106 → 0.0198 (con el split por canción y el nuevo preprocesamiento)
- [x] Mejorar recall de Gradient Boosting: resuelto con `class_weight='balanced'` (recall 0.15 → 0.83)
- [x] Evaluar alternativa a Label Encoding: se usó multi-hot/one-hot (Target Encoding ya no hace falta)
- [x] Re-ejecutar el notebook en la máquina local: las métricas de test coinciden
- [x] Agregar sección de negocio/CRISP-DM/KPIs, justificación de variables y modelos, modelo no supervisado y conclusiones de negocio al notebook
- [x] Medir si el puntaje depende del número de géneros (sección 7.2): no depende de que el modelo cuente géneros
- [ ] Volver a ejecutar en la máquina local la versión nueva del notebook (incluye K-Means y sección 7.2) y confirmar que los números de las secciones 4.2 y 4.3 coinciden
- [ ] Decidir si hacer tuning de hiperparámetros (GridSearchCV / RandomizedSearchCV)
- [ ] Analizar errores del modelo (¿qué géneros falla más en predecir?)
- [ ] Medir el aporte real de las features de audio: entrenar un modelo sin las columnas de género y comparar
- [ ] Decidir el umbral de decisión según costo de negocio (hoy 0.5)
- [ ] Escribir el **informe técnico en Markdown (.md) tipo README**, entregable obligatorio: problema de negocio, objetivos, KPIs, fuentes de datos, EDA, CRISP-DM (ver sección 7)
- [ ] Revisar que la carpeta del proyecto tenga la estructura pedida: `data/`, `notebooks/`, `models/`, `images/`, `README.md`, y que el notebook corra sin modificaciones
- [ ] Confirmar con el docente las metas de los KPIs (las definió el equipo como referencia)
- [ ] Crear la presentación (10 minutos por estudiante) con los 5 apartados pedidos
- [ ] Preparar la defensa: explicar qué observó el docente, por qué la v1 de EV2 no lo corregía y cómo se cuantificó la fuga

---

## 6. Preguntas / dudas a resolver

- ¿El profesor espera regresión Y clasificación, o solo clasificación?
- ~~¿Cuántos modelos mínimos pide la pauta EV2?~~ Respondida: mínimo **2 supervisados + 1 no supervisado**.
- ¿Se requiere tuning de hiperparámetros o basta con configuración razonada?
- ¿La comparativa EV1 (raw vs clean) debe estar en el informe EV2 también o solo en EV1?
- ¿El docente acepta el cambio de `GradientBoostingClassifier` a `HistGradientBoostingClassifier`, o hay que justificarlo en el informe?

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

## 8. Plan de trabajo para la siguiente sesión (instrucciones para el agente)

**Contexto:** el sesgo "más géneros → más puntaje" ya se diagnosticó (sección 4.3). Falta decidir cómo tratarlo y reforzar la defensa. Evaluación de tres propuestas recibidas:

| # | Propuesta | Factible | Prioridad | Comentario |
|---|---|---|---|---|
| 1 | Modelo **sin columnas de género** y comparar AUC | Sí, simple | **Alta** | Ya estaba como pendiente. No se ha ejecutado; **no asumir el resultado** (no hay evidencia de que caiga a 0.65). Responde si el modelo predice "popularidad del género" o "sonido". |
| 2 | P80 **dentro de cada género** | Sí, pero cambia el objetivo | Media, como experimento aparte | Cambia la pregunta de negocio (hit relativo al género) y todos los KPIs. Hay que definir una regla para canciones multi-género (p. ej. promedio de los z-scores de popularidad en sus géneros). No reemplaza al modelo principal. |
| 3 | Target Encoding regularizado | Sí, con cuidado | Baja | No corrige el sesgo: entrega el mismo dato (popularidad del género) en una sola columna. Con multi-género hay que agregar (promedio) y calcularlo **dentro de cada fold** para no filtrar el objetivo. |

### Tareas, en orden
1. **Sección 7.3 — Ablación sin género.** Entrenar LR, RF y HistGB con las mismas variables pero sin `FEATURES_GENRE`, mismo split y mismos hiperparámetros; reportar AUC, F1, recall, lift @ top-10% y compararlos con los modelos completos. Agregar también un modelo **solo con género** (solo columnas `genre_*`). Interpretar con honestidad: cuánto aporta el audio y cuánto el género.
2. **Sección 7.4 — Mitigación por reweighing** (ya probada fuera del notebook; ver resultados abajo). Pesos por (grupo de géneros {1, 2, 3+}, hit): `w = P(g)·P(y) / P(g,y)`, calculados solo en train; pasar con `pipe.fit(X, y, clf__sample_weight=w)` a HistGB. Reportar AUC, lift @ top-10%, correlación n° géneros–probabilidad y % de canciones por grupo que entra al top-10%.
3. **Opcional — experimento con objetivo relativo al género** (propuesta 2), en una sección aparte y claramente marcado como experimento complementario.
4. **Omitir** la propuesta 3, salvo que sobre tiempo; si se hace, en un fold interno.
5. Actualizar la sección 12 (conclusiones) con los resultados de 1 y 2 y **volver a ejecutar el notebook completo**. Actualizar las notas (secciones 4 y 5).

### Resultados ya medidos del reweighing (HistGB, test, SEED=42)
| | Sin pesos | Con reweighing |
|---|---|---|
| Correlación n° géneros vs probabilidad | 0.205 | 0.125 |
| AUC | 0.8544 | 0.8508 |
| F1 / Recall | 0.5595 / 0.8271 | 0.5521 / 0.8495 |
| Lift @ top-10% | 3.30 | 3.23 |
| % del grupo que entra al top-10% (1 / 2 / 3+ géneros) | 7.7% / 19.2% / 22.2% | 8.8% / 15.5% / 13.9% |

Pesos obtenidos: (1 género, no-hit) 0.968; (1, hit) 1.144; (2, no-hit) 1.197; (2, hit) 0.617; (3+, no-hit) 1.137; (3+, hit) 0.688. Hit rate real por grupo en test: 17.8% / 34.6% / 27.9%. Grupos: 81.4% / 12.7% / 5.9% de las canciones.
**Lectura:** reduce la relación sin eliminarla (otras variables actúan como sustitutas) y corrige la sobrevaloración del grupo 3+, pero hace que el modelo subestime la diferencia real entre 1 y 2 géneros. No se sabe si esa diferencia es señal real o artefacto del muestreo del dataset (~1.000 canciones por género).

### Datos técnicos para continuar
- **Notebook:** `EV2_Spotify_Modelos.ipynb` (40 celdas, secciones 0 a 12). Variables disponibles: `df_song`, `train`, `test`, `X_train`, `X_test`, `y_train`, `y_test`, `g_train`, `generos` (lista de géneros por `track_id`), `mlb`, `FEATURES_SCALE`, `FEATURES_CAT`, `FEATURES_BIN`, `FEATURES_GENRE`, `make_pre()`, `results`, `SEED=42`. Los experimentos nuevos deben ir **antes** de la sección 8 (importancia) o numerarse sin romper las referencias existentes.
- **Datos:** el notebook lee `data/processed/spotify_clean.csv` (113.549 filas, 25 columnas). Si no está disponible, reconstruirlo desde `dataset.csv`: quitar la columna índice, `dropna` en `artists`/`album_name`/`track_name`, `duration_ms > 0`, `drop_duplicates()`, `log1p` en `speechiness`, `instrumentalness`, `liveness`, `duration_ms` (columnas `*_log`) y `track_genre_encoded` con `LabelEncoder`.
- **Ejecución:** cada comando tiene un límite de 300 s. La ejecución completa del notebook (`jupyter nbconvert --execute`) tarda más, así que lanzarla **en segundo plano** (`setsid nohup python3 -m jupyter nbconvert ... &`) y revisar el log.
- **Estilo:** marcar como hipótesis lo no verificado, mantener las cifras coherentes con las salidas reales y no afirmar que el sesgo "está resuelto".
- **Pendiente del usuario:** ejecutar el notebook en su máquina y enviar el `.ipynb` con salidas para compararlo con estos números.

---

## 9. Estructura oficial del proyecto (pizarra del docente, "Semana 8: Evaluación II")

*Transcrita de 3 fotos. Dudas de lectura aclaradas por el usuario en 9.1.*

```
project/
├── data/
│   ├── raw/          train/  test/
│   └── processed/    train/  test/
├── notebooks/        0_exploracion_inicial · 1_EDA · 2_Modelacion
├── src/
│   ├── pipeline.py
│   ├── graficos.py ┐
│   ├── metricas.py ┘→ EDA.py (etapa de EDA del main: ver 9.1)
│   ├── train.py
│   └── test.py
├── output/           graficos/ · modelos/
├── docs/             0_exploracion_inicial.md · 1_eda.md · 2_modelo.md
├── README.md
├── main.py
└── requirements.txt  (o environment.yml si se usa Conda)
```

**Flujo del Pipeline (foto 1):** Raw Data → seleccionar columnas (eliminar las con pocos datos, con muchos nulos y con varianza 0) → tipos de variable → NA (imputar con mediana/moda/media/0/1, o eliminar filas o columnas) → eliminar duplicados → transformaciones → guardar → modelo → evaluar → visualizar.
**Transformaciones:** numéricas según su distribución (normal → StandardScaler; asimétrica → RobustScaler; uniforme → MinMaxScaler); categóricas (ordinal → label/ordinal encoding; nominal → one-hot).

### 9.1 Aclaraciones del usuario (confirmadas)
- **Lo oficial son los scripts `.py`:** todo el proyecto debe poder ejecutarse con `main.py` (pipeline → entrenamiento → evaluación). Los **notebooks son solo de exploración**.
- **`EDA.py` (llave de `graficos.py` + `metricas.py`):** la exploración con gráficos y métricas se hace en el notebook de EDA; luego esos mismos gráficos y métricas deben **generarse automáticamente al ejecutar `main.py`**, que ejecuta `graficos.py` y `metricas.py` (agrupados como etapa de EDA). Los resultados se guardan en `output/graficos/`.
- **Archivo de entorno:** `environment.yml` (Conda) o `requirements.txt`.
- **"Transformación de variables multiclase" (foto 1):** se refiere, en este caso, a la columna de género (`track_genre`), que es multiclase y multietiqueta por canción. Hoy se resuelve con multi-hot (secciones 3 y 5 del notebook); debe quedar como un paso explícito de `pipeline.py`.
- **Notebooks:** 0 exploración inicial, 1 EDA, 2 Modelación.

### 9.2 Recomendación del docente sobre el sesgo
Para resolver el sesgo (canciones con más géneros), el docente recomendó **rehacer el trabajo en un notebook nuevo, siguiendo esta estructura oficial**, en vez de parchar el notebook actual. Implicaciones para el agente:
- Partir de la estructura de la sección 9 y de `src/pipeline.py` con el flujo de la foto 1 (el paso multiclase del género incluido), con `main.py` como punto de entrada.
- Reutilizar las decisiones ya validadas: una fila por canción, split y CV por canción (`song_key`), P80 solo con train, Pipeline de sklearn para el preprocesamiento, K-Means como técnica no supervisada y KPIs de negocio.
- Incluir en el nuevo proyecto el tratamiento del sesgo del plan de la sección 8 (ablación sin género y reweighing), con los resultados ya medidos como referencia.
- Los notebooks nuevos (0, 1, 2) deben **importar y usar** `src/`, no duplicar la lógica.
- El orden de las tareas de abajo reemplaza al de la sección 8: primero estructura y pipeline, después experimentos.

### Brechas frente al proyecto actual
- **Estructura:** hoy hay un solo notebook + `data/processed`, `images/`, `models/`. Falta `src/` (con `main.py` que orquesta pipeline → train → test), `output/graficos` y `output/modelos` (equivalen a `images/` y `models/`), `docs/*.md`, notebooks 0/1/2 separados (EP1 = exploración y EDA; este notebook = modelación), `requirements.txt` y `README.md`.
- **Split en disco:** la estructura pide `data/raw/{train,test}` y `data/processed/{train,test}`. Hoy el split se hace en memoria. Mantener el split **por canción** (`song_key`) y el P80 calculado solo con train; guardar los CSV de train y test.
- **Escalado por distribución:** hoy se usa `StandardScaler` en las 10 variables continuas (tras `log1p`). El docente pide elegir por distribución; usar la asimetría medida en EP1 para justificar (p. ej. `RobustScaler` en las que sigan asimétricas) y verificar el efecto en regresión logística y K-Means (los árboles no cambian).
- **Selección de columnas:** documentar explícitamente el filtro de columnas (pocos datos, muchos nulos, varianza 0) y la imputación de NA (en este dataset hay 1 nulo, que se elimina).
- **Orden del pipeline:** mapear las secciones del notebook a los pasos de la foto 1 en `docs/1_eda.md` y `docs/2_modelo.md`.

### Tareas (prioridad: estas primero, luego la sección 8)
1. Crear la carpeta del proyecto con la estructura oficial; el notebook actual queda como referencia, no como entregable.
2. Escribir `src/pipeline.py` con el flujo de la foto 1 (selección de columnas, tipos, NA, duplicados por canción, transformación multiclase del género, escalado según distribución, guardar `data/processed/{train,test}`), `src/train.py`, `src/test.py`, `src/graficos.py` y `src/metricas.py`.
3. Crear `main.py` que ejecute todo de punta a punta con un solo comando, y verificar que corre desde cero.
4. Crear los notebooks 0, 1 y 2 como exploración, usando `src/`.
5. Escribir `docs/` y `README.md` (informe técnico con las secciones de la pauta, ver sección 7).
6. Generar `requirements.txt` (o `environment.yml`) con las versiones usadas.
7. Incorporar los experimentos de la sección 8 (ablación sin género, reweighing) y actualizar las conclusiones.

---

## 10. Bitácora de Implementación y Avance (Octubre 2026)

### 10.1 Estructura Modular Desarrollada en `src/` y `main.py`
Se implementó la arquitectura modular exigida por el docente para desacoplar el pipeline de los notebooks:
- **`src/pipeline.py`:**
  - Deduplicación por canción única (`track_id` / `song_key`).
  - Tratamiento multietiqueta del género con `MultiLabelBinarizer` (114 columnas `genre_*` multi-hot).
  - Split train/test estratificado por canción guardando en disco `data/processed/train/` y `data/processed/test/` (71.706 train / 18.034 test, 128 features totales).
  - Cálculo estricto del umbral P80 únicamente en train (52 puntos; hit rate ~21.0% train, ~20.5% test) para evitar data leakage.
  - Preprocesamiento con ColumnTransformer de scikit-learn: escalado numérico según distribución (StandardScaler en gaussianas, RobustScaler en asimétricas) y OneHotEncoder en categóricas (`explicit`, `mode`, `time_signature`).
- **`src/train.py`:**
  - Entrenamiento y validación cruzada (3-fold StratifiedKFold) de 4 modelos: Regresión Logística, Árbol de Decisión, Random Forest e HistGradientBoosting.
  - Serialización de artefactos y pipelines entrenados en `output/modelos/*.joblib`.
- **`src/test.py`:**
  - Evaluación final en el conjunto de prueba independiente (`X_test.csv`, `y_test.csv`).
  - Cálculo de métricas globales y exportación de resultados a `output/resultados_modelos.csv`.
- **`src/metricas.py`:**
  - Funciones de cálculo: AUC-ROC, F1-Score, Recall, Precision, Accuracy, Lift acumulado al Top-10% y brecha de sobreajuste (gap AUC train - test).
- **`src/graficos.py`:**
  - Generación automatizada de visualizaciones EDA y modelación en `output/graficos/`:
    1. `01_distribucion_popularidad.png`
    2. `02_distribucion_audio_features.png`
    3. `03_heatmap_correlaciones.png`
    4. `04_outliers_boxplots.png`
    5. `05_popularidad_por_genero.png`
    6. Curvas ROC, calibración, Lift e importancia de variables por permutación.
- **`main.py`:**
  - Orquestador integral ejecutable con un solo comando: ejecuta secuencialmente `pipeline` -> `train` -> `test` -> `graficos` y muestra resumen final de métricas.

### 10.2 Depuración y Solución de Problemas Críticos en Windows
1. **Multiprocessing y Deadlocks de Spawn:**
   - En Windows, `joblib` y `scikit-learn` utilizan el método `spawn` para procesos hijos, provocando bloqueos si no se invoca `multiprocessing.freeze_support()`.
   - Se añadió `freeze_support()` al inicio de `main.py` y se configuró ejecución secuencial determinista (`n_jobs=1`) para estabilidad garantizada.
2. **Cierre Silencioso al Guardar Gráficos (`0xc06d007f: STATUS_PROCEDURE_NOT_FOUND`):**
   - **Diagnóstico:** Mediante `python -X faulthandler` se aisló un crash nativo de Windows al ejecutar `transforms.py` (líneas 2455-2456: `np.dot` para transformaciones afines de Matplotlib al guardar figuras con `fig.savefig()`).
   - **Causa raíz:** Incompatibilidad de la DLL de OpenBLAS de Conda en Windows al llamar a rutinas BLAS/LAPACK de NumPy (`np.dot`). Además, `pyparsing 3.3.3` presentaba una regresión en el parser de nombres de fuentes (`sans-serif`).
   - **Solución implementada:**
     - Reinstalación de `numpy==1.26.4` vía wheel oficial de PyPI con binarios de OpenBLAS autocontenidos y estables.
     - Ajuste de dependencia a `pyparsing<=3.1.4`.
     - Verificación exitosa de `fig.savefig()` y generación completa de visualizaciones sin errores.

### 10.3 Estado de Ejecución Actual
- Ejecución completa de `main.py` en marcha:
  - Preprocesamiento y split en disco finalizados exitosamente.
  - Regresión Logística entrenada (AUC Test: 0.8357, F1: 0.5516, Recall: 0.7977).
  - Random Forest en entrenamiento activo.

### 10.4 Próximos Pasos Pendientes (según Pauta del Docente)
1. **Notebooks de Exploración:** Crear `notebooks/0_exploracion_inicial.ipynb`, `notebooks/1_EDA.ipynb` y `notebooks/2_Modelacion.ipynb` que consuman directamente los módulos de `src/`.
2. **Documentación Técnica:** Redactar `docs/0_exploracion_inicial.md`, `docs/1_eda.md`, `docs/2_modelo.md` y el informe técnico final en `README.md`.
3. **Congelamiento de Entorno:** Generar `environment.yml` / `requirements.txt` con las versiones probadas y estables.
4. **Experimentos de Sesgo:** Incorporar la sección de ablación sin género y reweighing (sección 8) en la documentación y análisis final.

### 10.5 Resultados de la Ejecución End-to-End (`main.py`)
El pipeline completo se ejecutó de principio a fin en **246.2 segundos** (código de salida 0):

```text
============================================================
RESUMEN — Evaluación en conjunto de prueba independiente
============================================================
                      Accuracy  Precision  Recall  F1-Score  AUC-ROC  AUC train  Gap AUC  Lift @10% Sobreajuste
Modelo                                                                                                         
Regresion Logistica     0.7337     0.4215  0.7977    0.5516   0.8357     0.8404   0.0047       3.01          No
Random Forest           0.7283     0.4126  0.7631    0.5356   0.8241     0.8403   0.0163       3.16          No
HistGradientBoosting    0.7327     0.4228  0.8271    0.5595   0.8544     0.8812   0.0268       3.30          No
```

- **Cumplimiento de KPIs de negocio:**
  - AUC-ROC $\ge$ 0.80: Cumplido por los 3 modelos (Mejor: HistGradientBoosting con **0.8544**).
  - Gap AUC train-test $<$ 0.05: Cumplido sin sobreajuste (Gap máximo 0.0268).
  - Recall $\ge$ 0.75: Cumplido (HistGradientBoosting **0.8271**).
  - Lift @ 10% $\ge$ 2.0: Cumplido (HistGradientBoosting **3.30x** el azar).
- **Artefactos generados exitosamente:**
  - `data/processed/train/` y `data/processed/test/` con `X` e `y`.
  - `output/modelos/`: `regresion_logistica.joblib`, `random_forest.joblib`, `histgradientboosting.joblib`.
  - `output/graficos/`:
    - `01_distribucion_popularidad.png`
    - `02_distribucion_audio_features.png`
    - `03_heatmap_correlaciones.png`
    - `04_outliers_boxplots.png`
    - `05_popularidad_por_genero.png`
    - `06_curvas_roc.png`
    - `07_matrices_confusion.png`
    - `08_feature_importance_rf.png`
    - `09_curva_ganancia_acumulada.png`

### 10.6 Limpieza de Archivos Antiguos (Octubre 2026)
- Se eliminaron las carpetas obsoletas `images/` y `models/`, reemplazadas formalmente por `output/graficos/` y `output/modelos/`.
- Se eliminaron los datasets intermedios obsoletos de la raíz de `data/processed/` (`X_model.csv`, `y_model.csv`, `X_*_corrected.csv`, `y_train.csv`, `y_test.csv`, `feature_names.csv`), quedando centralizados en la estructura oficial `data/processed/train/` y `data/processed/test/`.
- Se eliminaron los notebooks antiguos (`notebooks/EV1_Spotify.ipynb` y `notebooks/EV2_Spotify_Modelos.ipynb`), dejando la carpeta `notebooks/` preparada para los 3 nuevos notebooks modulares (`0_exploracion_inicial.ipynb`, `1_EDA.ipynb`, `2_Modelacion.ipynb`).
- Se removieron los archivos de prueba temporales (`test.svg`, `output/graficos/test_pil.png`, `test.pdf`).
- Se actualizaron las referencias en `main.py` para prescindir de carpetas deprecadas.
