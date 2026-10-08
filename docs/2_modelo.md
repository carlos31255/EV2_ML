# 2. Modelación

**Notebook:** `notebooks/2_Modelacion.ipynb` · **Código:** `src/train.py`, `src/test.py`, `src/metricas.py`, `src/graficos.py`

## 1. Qué hace este paso

Consume `data/processed/train|test` (generado al cierre del EDA; **no divide ni transforma datos**), entrena y evalúa los modelos, y complementa los resultados con análisis de sesgos. `verificar_processed()` avisa si falta `data/processed`.

## 2. Modelos y protocolo

| Modelo | Tipo |
|---|---|
| Regresión Logística | Supervisado (lineal) |
| Random Forest | Supervisado (árboles, `max_depth=15`, `min_samples_leaf=10`) |
| HistGradientBoosting | Supervisado (boosting, `max_iter=300`, `learning_rate=0.05`, `max_depth=6`) |
| K-Means | No supervisado (solo variables de audio) |

- **Preprocesamiento dentro del `Pipeline`:** `StandardScaler` (numéricas), `OneHotEncoder` (`key`, `time_signature`), `passthrough` (`explicit`, `mode` y las 114 columnas `genre_*`).
- **Desbalance:** `class_weight="balanced"` en los tres modelos supervisados.
- **Validación cruzada:** 5 folds estratificados y **agrupados por canción** (`song_key`), semilla 42.
- **Selección del mejor modelo:** por AUC de validación cruzada sobre train; **test se usa solo para el reporte final**.
- **Umbral de decisión:** 0,5 (por defecto, no se ajustó; ver sección 5).

## 3. Resultados

### Validación cruzada (train)

| Modelo | AUC CV | Gap CV | Recall CV | Precisión CV | F1 CV |
|---|---|---|---|---|---|
| Regresión Logística | 0,8380 | 0,0026 | 0,806 | 0,434 | 0,564 |
| Random Forest | 0,8217 | 0,0203 | 0,738 | 0,425 | 0,539 |
| **HistGradientBoosting** | **0,8561** | 0,0284 | 0,825 | 0,434 | 0,569 |

**Mejor modelo: HistGradientBoosting** (AUC de CV más alto).

### Test (18.034 canciones, hit rate 0,205)

| Modelo | AUC | PR-AUC | Precisión | Recall | F1 | Gap AUC | Lift@10% |
|---|---|---|---|---|---|---|---|
| Regresión Logística | 0,8357 | 0,562 | 0,422 | 0,798 | 0,552 | 0,0047 | 3,01 |
| Random Forest | 0,8255 | 0,574 | 0,417 | 0,757 | 0,538 | 0,0153 | 3,19 |
| **HistGradientBoosting** | **0,8569** | **0,619** | 0,425 | **0,833** | 0,563 | 0,0255 | **3,36** |

El AUC de test del mejor modelo (0,8569) es casi igual al de CV (0,8561): no hay señal de que la selección del modelo haya inflado el resultado.

### KPIs de negocio

| KPI | Meta | HGB | Regresión Logística | Random Forest |
|---|---|---|---|---|
| AUC-ROC | ≥ 0,80 | 0,857 ✔ | 0,836 ✔ | 0,826 ✔ |
| Gap AUC | < 0,05 | 0,026 ✔ | 0,005 ✔ | 0,015 ✔ |
| Recall | ≥ 0,75 | 0,833 ✔ | 0,798 ✔ | 0,757 ✔ (al límite) |
| Lift@10% | ≥ 2,0 | 3,36 ✔ | 3,01 ✔ | 3,19 ✔ |

> **Nota:** el recall de Random Forest está al límite. En otra corrida con un split ligeramente distinto fue 0,723, y con el multi-hot normalizado cae a 0,735. Debe presentarse como "cerca del límite", no como cumplimiento firme.

## 4. Lo que el modelo aprende: el género

| Prueba | Resultado |
|---|---|
| **Ablación** (HGB, CV agrupada) | AUC **0,856 con género → 0,695 sin género** (aporte +0,162) |
| **Importancia por permutación** (test) | Bloque de género: caída de AUC **0,303**; siguiente variable (`instrumentalness_log`): 0,009 |
| **Solo audio** (test, columnas de género anuladas) | AUC 0,693 · recall 0,687 · precisión 0,304 · **lift 1,96 (bajo el KPI de 2,0)** |
| **K-Means** (solo audio, k=2, silhouette 0,240) | Los clusters separan estilos (acústico/tranquilo frente a enérgico), pero su hit rate es casi igual: 0,191 frente a 0,210 |

**Lectura:** el KPI de AUC se cumple **gracias al género**. Con solo audio no se alcanza el 0,80.

## 5. Umbral de decisión (HGB, test)

| Umbral | Precisión | Recall | F1 | % marcadas como hit |
|---|---|---|---|---|
| 0,3 | 0,315 | 0,967 | 0,475 | 63,1% |
| 0,4 | 0,354 | 0,925 | 0,512 | 53,6% |
| **0,5** | **0,425** | **0,833** | **0,563** | **40,2%** |
| 0,6 | 0,559 | 0,613 | 0,585 | 22,5% |
| 0,7 | 0,664 | 0,387 | 0,489 | 12,0% |
| 0,8 | 0,787 | 0,185 | 0,300 | 4,8% |

**Decisión:** se mantiene 0,5. Con 0,6 el F1 es el mejor (0,585), pero el recall (0,613) deja de cumplir el KPI de 0,75. Con 0,5 se cumple el KPI marcando el 40% de las canciones, con una precisión de ≈ 0,43: **más de la mitad de las canciones marcadas no son hits**, un costo a considerar en campañas.

## 6. Recall por género

La versión inicial (géneros con ≥ 20 hits: 74 de 114) deja fuera los géneros de pocos hits, así que es optimista (mediana 0,87). Con **todos** los géneros con al menos un hit (109 de 114):

- Recall promedio por género (macro): **0,580** (frente a 0,846 global por género-canción).
- **56 de 109 géneros (51%) no llegan al KPI de 0,75.**
- Correlación entre el hit rate del género y su recall: **0,94**.

| Hit rate del género | Géneros | Recall | Precisión | % de canciones marcadas |
|---|---|---|---|---|
| Bajo (< 10%) | 35 | 0,148 | 0,191 | 2,9% |
| Medio (10–25%) | 31 | 0,661 | 0,287 | 42,3% |
| Alto (25–50%) | 33 | 0,931 | 0,430 | 81,0% |
| Muy alto (> 50%) | 10 | 0,990 | 0,673 | 92,9% |

**Lectura:** el modelo se comporta en gran parte como un filtro de popularidad del género. Su recall global se sostiene porque en géneros con muchos hits marca casi todo, y en géneros de pocos hits casi no detecta. (En el tramo "bajo" hay ≈ 7 hits por género: usar las cifras por tramo, no por género individual.)

## 7. Pruebas de sensibilidad y decisión final

Cada prueba reentrena el mejor modelo (mismos hiperparámetros) cambiando solo un aspecto.

| # | Alternativa probada | Resultado | Decisión |
|---|---|---|---|
| 1 | **Split agrupado por artista principal** (3 semillas) en lugar de por canción | AUC 0,856 → **0,845** (−0,011), recall 0,832 → 0,812, lift 3,36 → 3,23; artistas de test vistos en train: 92% → 36% | **Se mantiene el split por canción.** El optimismo por artistas repetidos es pequeño y los KPIs se mantienen; se declara como limitación |
| 2 | **Excluir popularidad 0** (train y test; P80 pasa a 54) | AUC con género 0,857 → 0,880; solo audio 0,693 → 0,711; aporte del género +0,164 → **+0,169** | **Se mantiene popularidad 0 como "no hit".** Excluirla mejora las métricas (menos ruido) pero no reduce el dominio del género; no explica el sesgo |
| 3 | **Target relativo al género** (hit = popularidad ≥ P80 del género; hit rate 0,198) | AUC con género 0,657 · solo audio 0,578 · lift 1,89 / 1,44. Disparidad de recall entre géneros baja (desv. estándar 0,389 → 0,175; correlación con hit rate 0,94 → 0,24) pero detecta peor (77% de géneros bajo 0,75) | **No se adopta como modelo final:** cambia el problema de negocio ("¿hit dentro de su género?") y rinde mucho peor. Se usa como medida de la señal del audio sin género (AUC ≈ 0,58–0,66) |
| 4 | **Multi-hot normalizado** (cada fila suma 1) en lugar de binario | AUC +0,002 en los tres modelos (LR 0,8377 · RF 0,8274 · HGB 0,8589); recall RF 0,757 → **0,735**; canciones de 2 géneros marcadas 57% → 60% | **No se adopta:** ganancia dentro del ruido, no cambia el patrón, empeora un KPI y obliga a re-ejecutar todo. Se mantiene **multi-hot binario** |
| 5 | **Entrenar y evaluar solo con canciones de 1 género** | AUC 0,863 | Evidencia de que el desempeño no depende del número de géneros |

### ¿El modelo premia a las canciones con más géneros?

El multi-hot lleva implícito el número de géneros (la suma de unos por fila), aunque `n_generos` no sea una feature. Resultados:

- **Predicción:** el modelo marca el 36% de las canciones de 1 género, el 57% de las de 2 y el 63% de las de 3 o más (hit rate real 0,178 · 0,346 · 0,279).
- **Esperado solo por sus géneros** (tasa de hit de cada género, calculada en train): 0,195 · 0,294 · 0,312. Solo las canciones de exactamente 2 géneros superan lo esperado (≈ +5 puntos).
- **AUC por subgrupo:** 0,865 en canciones de 1 género y 0,804 en las de 2 o más.
- **Neutralizando el conteo** (pruebas 4 y 5): el AUC no baja.

**Conclusión:** el modelo marca más canciones multigénero porque suelen pertenecer a géneros de hit rate alto (efecto de composición), no porque el número de géneros sostenga el resultado.

### Alternativas descartadas por diseño (no probadas)

- Selección del modelo con test: descartado para evitar optimismo (se usa CV agrupada).
- Escalado en `processed`: descartado (filtra información entre folds).
- Umbral de decisión por género calibrado con train: **no se probó**; es la mitigación más simple pendiente (emparejaría el recall entre géneros a costa de la precisión).

### Decisión final

**Modelo: HistGradientBoosting, multi-hot binario, split agrupado por canción, target absoluto (P80 de train = 52), popularidad 0 como no hit, umbral 0,5.**

Se eligió porque: (a) tiene el mejor AUC de CV y de test y cumple los cuatro KPIs; (b) ninguna alternativa probada mejoró el resultado de forma relevante ni corrigió el sesgo de género sin un costo mayor (pérdida de desempeño o cambio del problema); (c) las diferencias medidas (artistas, popularidad 0, número de géneros) son pequeñas o no cambian la conclusión.

## 8. Sesgos: balance final

| Sesgo | Estado |
|---|---|
| Duplicados, fuga train/test, selección con test | **Evitados** |
| Artistas repetidos, popularidad 0, número de géneros | **Medidos, efecto pequeño** (se declaran) |
| **Dominio del género** | **No evitado**: sin género el AUC cae a 0,69 |
| **Recall desigual entre géneros** | **No evitado**: 0,15 en géneros de hit rate bajo frente a 0,99 en muy alto |
| Muestreo 1.000 filas por género | No corregible |

## 9. Limitaciones y uso recomendado

1. El KPI de AUC ≥ 0,80 depende del género; con solo audio el AUC es ≈ 0,69 y el lift 1,96.
2. El recall es muy desigual entre géneros; el 51% de los géneros no llega a 0,75.
3. La precisión es ≈ 0,43: más de la mitad de las canciones marcadas no son hits.
4. El split por canción deja artistas repetidos entre train y test (efecto ≈ 0,01 de AUC).
5. Popularidad 0 concentrada en géneros y artistas; puede ser falta de dato.
6. El recall de Random Forest está al límite del KPI y depende del split.
7. El dataset (1.000 filas por género, popularidad puntual) no permite extrapolar al mercado real.

**Uso recomendado:** priorizar canciones para campañas (lift 3,36 en el 10% superior). El modelo captura sobre todo qué géneros son populares, **no** qué hace a una canción un hit; no debe usarse para afirmar que el audio de una canción causará su popularidad.

## 10. Salidas

- **Modelos:** `output/modelos/` (`regresion_logistica.joblib`, `random_forest.joblib`, `histgradientboosting.joblib`, `best_model.joblib`) y tablas de resultados (`resultados_cv.csv`, `resultados_test.csv`, `kpis_cumplidos.csv`, `ablacion_genero.csv`).
- **Gráficos** (`output/graficos/`): `08_curvas_roc`, `09_curvas_pr`, `13_recall_por_genero`, `14_importancia_permutacion`, `15_kmeans_silhouette`, `16_recall_vs_hit_rate_genero`.

## 11. Reproducción

```
python main.py
```
ejecuta split raw → processed → entrenamiento → evaluación en test. Para regenerar todo desde cero, borrar `data/raw/train|test`, `data/processed` y `output/modelos` antes. El split es determinista (`random_state=42`) y produjo los mismos resultados en dos entornos distintos.
