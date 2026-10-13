# Informe Técnico: Predicción de Popularidad en Spotify

## 1. Problema de Negocio
La industria musical enfrenta el constante desafío de identificar qué canciones alcanzarán el éxito comercial para optimizar la asignación de presupuestos de promoción y marketing. El presente proyecto aborda este desafío mediante un modelo de Machine Learning cuyo objetivo es **predecir si una canción en Spotify será un "hit" comercial** utilizando exclusivamente sus características intrínsecas de audio y su género musical. 

La pregunta de negocio central es: *¿Debe un sello discográfico invertir presupuesto de promoción en esta canción?* Para responderla, se define un "hit" como toda canción cuya popularidad alcance o supere el **percentil 80** del conjunto de datos (P80), lo que en este estudio equivale a un umbral de 52 puntos de popularidad (calculado estrictamente sobre el conjunto de entrenamiento).

## 2. KPIs de Negocio
Para asegurar que el modelo generado sea verdaderamente valioso para las decisiones del sello discográfico, se establecieron cuatro métricas críticas (KPIs):
1. **AUC-ROC $\ge$ 0.80:** Garantiza una alta capacidad del modelo para distinguir consistentemente entre un hit y un no-hit.
2. **Gap AUC train-test $<$ 0.05:** Control estricto de sobreajuste (overfitting) para asegurar que el modelo generalice en nuevas canciones.
3. **Recall $\ge$ 0.75:** Prioriza capturar la mayor cantidad de hits posibles (minimizando los falsos negativos, que equivaldrían a perder oportunidades de oro).
4. **Lift @ 10% $\ge$ 2.0:** Mide el retorno de inversión en marketing; al enfocarse en el top 10% de predicciones más altas, se espera encontrar al menos el doble de hits que si se seleccionaran canciones al azar.

## 3. Fuentes de Datos
El proyecto utiliza el **Spotify Tracks Dataset** proveniente de HuggingFace, un catálogo masivo que contiene metadatos y variables acústicas obtenidas mediante la API oficial de Spotify. Originalmente consta de 114.000 filas. Tras una rigurosa deduplicación (fusionando copias y consolidando atributos), se obtuvo un catálogo de **89.740 canciones únicas**.

## 4. Metodología CRISP-DM y Análisis Exploratorio (EDA)
El proyecto se estructuró bajo el marco metodológico CRISP-DM, asegurando un flujo de trabajo robusto y libre de fugas de información.

### 4.1. Análisis Exploratorio (EDA)
El EDA reveló características clave de los datos:
- **Asimetría Acústica:** Variables como `duration_ms`, `speechiness` y `liveness` presentaban colas largas. Se aplicó una transformación logarítmica (`log1p`) para normalizar estas distribuciones.
- **La incógnita del "Popularidad 0":** El 10.3% del dataset registra popularidad cero. Este bloque no es una cola estadística natural, sino un artefacto concentrado en géneros específicos (jazz, iranian, romance). Se decidió conservarlo bajo la etiqueta de "no-hit".
- **Débil señal de audio:** Las correlaciones lineales directas entre variables acústicas puras (como `instrumentalness` o `liveness`) y la popularidad resultaron débiles (correlaciones cercanas a -0.09).

### 4.2. Preparación de Datos (Data Preparation)
- **Codificación de Género:** Al encontrar que el 18.1% de las canciones poseía múltiples géneros en distintas entradas, se diseñó una lógica de consolidación agrupada por `track_id` y se aplicó *Multi-Hot Encoding*, generando 114 columnas independientes de género sin perder dimensionalidad musical.
- **Split Robusto (GroupShuffleSplit):** Para prevenir la fuga de datos por versiones de una misma canción, el conjunto se dividió agrupando por la clave única `song_key` (Artista + Título). El split final resultó en 71.706 canciones para entrenamiento (Train) y 18.034 para prueba (Test).

## 5. Justificación de Algoritmos Supervisados
Con el fin de lograr el mejor rendimiento predictivo, se evaluaron tres algoritmos de clasificación de distinta complejidad:
1. **Regresión Logística:** Seleccionada como modelo base (*baseline*) lineal. Es altamente interpretable y rápido, ideal para comprender la dirección de los pesos asignados a cada variable.
2. **Random Forest:** Un ensamble en paralelo basado en árboles de decisión. Se justifica por su capacidad para capturar relaciones no lineales complejas sin ser excesivamente sensible a hiperparámetros o outliers.
3. **HistGradientBoosting:** Un algoritmo de *boosting* secuencial optimizado (similar a LightGBM). Seleccionado por su eficiencia en datasets estructurados medianos y grandes, y su habilidad excepcional para capturar interacciones intrincadas entre el audio y el género musical.

## 6. Modelación No Supervisada: K-Means
Para comprender si las características de audio por sí solas pueden agrupar éxitos comerciales, se aplicó un clustering con K-Means utilizando **únicamente las variables acústicas** (excluyendo género y popularidad). 
Al medir el *hit rate* promedio dentro de cada cluster, las diferencias fueron mínimas. Esto valida un hallazgo fundamental del EDA: la señal de audio pura no basta para discriminar éxitos; el contexto (el género) es el motor que determina la tracción comercial de una pista.

## 7. Interpretación de Negocio y Resultados Finales
El modelo **HistGradientBoosting** superó a sus pares en las validaciones y fue seleccionado definitivamente tras cumplir y exceder los cuatro KPIs en el conjunto de prueba (18.034 canciones).

### 7.1. Tabla Oficial de Resultados (Conjunto de Prueba)

| Modelo | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Lift @10% | Gap AUC |
|---|---|---|---|---|---|---|---|
| Regresión Logística | 0.7338 | 0.4216 | 0.7977 | 0.5516 | 0.8357 | 3.01 | 0.0047 |
| Random Forest | 0.7331 | 0.4173 | 0.7574 | 0.5381 | 0.8255 | 3.19 | 0.0153 |
| **HistGradientBoosting** | **0.7347** | **0.4254** | **0.8331** | **0.5632** | **0.8569** | **3.36** | **0.0255** |

*El modelo HistGradientBoosting logró el mejor Recall y AUC-ROC. Gracias al uso de pesos balanceados (`class_weight="balanced"`), prioriza encontrar éxitos reales a expensas de la precisión.*

### 7.2. Ablación de Género y Sesgos
El peso del género musical domina las predicciones. Un análisis de ablación (entrenar el modelo con y sin las 114 variables de género) demostró este punto:
- **AUC CV (Train) con género:** 0.8561
- **AUC CV (Train) sin género:** 0.6946
- **Aporte neto del género:** +0.1615 AUC
El modelo captura un *recall* alto en los géneros dominantes del mercado, alineándose a cómo funciona realmente la distribución de popularidad masiva en la plataforma. Sin embargo, esto genera una disparidad notable: a nivel individual, **56 de los 109 géneros evaluados (51%) no alcanzan el KPI de 0.75**, concentrándose el acierto en géneros masivos (recall > 0.90) frente a géneros de nicho (recall ~0.15).

### 7.3. Trade-Off de Umbral
El modelo por defecto evalúa el umbral de decisión en el punto 0.50, lo que maximiza el Recall (0.833) para no perder casi ningún hit potencial. Sin embargo, para un sello con presupuesto limitado de marketing, este umbral puede subirse artificialmente. Esto sacrificará algo de Recall, pero elevará drásticamente la Precisión (es decir, cada dólar invertido tendrá mayor certeza de ir a un hit seguro).

## 8. Limitaciones y Consideraciones Finales
Para interpretar correctamente el modelo en un entorno productivo real, deben considerarse las siguientes limitaciones detectadas:
1. **Split por canción y no por artista:** El 72.1% de las canciones de prueba comparte artista con las de entrenamiento. Dado que los artistas consolidados tienden a tener catálogos exitosos completos (o viceversa), las métricas de prueba presentan cierto sesgo optimista derivado del reconocimiento del artista.
2. **Ambigüedad de versiones:** Se identificaron más de 1.000 canciones que son hits en una versión (ej. radio edit) pero no hits en otra (álbum version).
3. **Muestreo por género no natural:** El dataset provee la misma cantidad original de pistas por género, lo cual no refleja la penetración real del mercado en Spotify. Las proyecciones deben normalizarse por volumen real de audiencia antes de dictar una estrategia de inversión.
4. **Dominancia del Género:** Al retirar el género, el AUC de 0.69 demuestra que la estructura acústica pura aporta apenas una ventaja marginal sobre el azar. Invertir solo basándose en un buen "sonido", ignorando la etiqueta comercial, es matemáticamente riesgoso.
