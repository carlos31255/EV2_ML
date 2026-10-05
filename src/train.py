"""
src/train.py
------------
Entrena los tres modelos supervisados (Pipeline sklearn) sobre los datos
procesados y guarda los pipelines en output/modelos/.

Modelos:
  1. Regresion Logistica  — baseline interpretable
  2. Random Forest        — ensamble principal
  3. HistGradientBoosting — ensamble avanzado (mejor AUC en EV2)

Preprocesamiento (dentro del Pipeline, sin fuga):
  - StandardScaler en variables continuas
  - OneHotEncoder en variables categoricas nominales
  - Variables binarias y genre_* sin transformar

Uso:
  python -m src.train
  from src.train import run
"""

import os
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_validate

from src.pipeline import FEATURES_SCALE, FEATURES_CAT, FEATURES_BIN
from src.metricas import evaluar_modelo, tabla_comparativa, verificar_kpis

# ── Constantes ────────────────────────────────────────────────────────────────

TRAIN_DIR   = os.path.join("data", "processed", "train")
TEST_DIR    = os.path.join("data", "processed", "test")
OUTPUT_DIR  = os.path.join("output", "modelos")
SEED        = 42
CV_FOLDS    = 5


# ── Preprocesador ─────────────────────────────────────────────────────────────

def make_preprocessor(features_genre: list) -> ColumnTransformer:
    """
    Construye el ColumnTransformer para el Pipeline:
      - Continuas  -> StandardScaler
      - Nominales  -> OneHotEncoder (handle_unknown='ignore')
      - Binarias + genre_*  -> passthrough
    """
    return ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), FEATURES_SCALE),
            ("ohe",   OneHotEncoder(handle_unknown="ignore", sparse_output=False), FEATURES_CAT),
            ("pass",  "passthrough", FEATURES_BIN + features_genre),
        ],
        remainder="drop"
    )


def definir_modelos(preprocessor: ColumnTransformer) -> dict:
    """
    Devuelve los tres Pipelines con sus hiperparametros justificados.
    Justificacion de la configuracion:
      - class_weight='balanced': dataset desbalanceado ~80/20 (hit/no-hit)
      - LR max_iter=2000:        convergencia con multi-hot (128 features)
      - RF max_depth=15, min_samples_leaf=10: limitar sobreajuste (gap v1=0.106)
      - HistGB lr=0.05, max_iter=300, max_depth=6: menor tasa de aprendizaje
                                                    para estabilidad
    """
    return {
        "Regresion Logistica": Pipeline([
            ("pre", preprocessor),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                       random_state=SEED, solver="saga", n_jobs=1))
        ]),
        "Random Forest": Pipeline([
            ("pre", preprocessor),
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_leaf=10,
                class_weight="balanced", random_state=SEED, n_jobs=1))
        ]),
        "HistGradientBoosting": Pipeline([
            ("pre", preprocessor),
            ("clf", HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05, max_depth=6,
                class_weight="balanced", random_state=SEED))
        ]),
    }


# ── Entrenamiento ─────────────────────────────────────────────────────────────

def run() -> dict:
    """
    Carga los datos procesados, entrena los modelos con CV y evaluacion final
    en test, guarda los pipelines y retorna los resultados.
    """
    print("=" * 60)
    print("TRAIN — Entrenamiento de modelos supervisados")
    print("=" * 60)

    # Cargar datos
    feat_names = pd.read_csv(os.path.join("data", "processed", "feature_names.csv"))
    features_genre = [f for f in feat_names["feature"] if f.startswith("genre_")]

    X_train = pd.read_csv(os.path.join(TRAIN_DIR, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(TRAIN_DIR, "y_train.csv")).squeeze()
    g_train = pd.read_csv(os.path.join(TRAIN_DIR, "g_train.csv")).squeeze()
    X_test  = pd.read_csv(os.path.join(TEST_DIR,  "X_test.csv"))
    y_test  = pd.read_csv(os.path.join(TEST_DIR,  "y_test.csv")).squeeze()

    print(f"[train] X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"[train] Hit rate train: {y_train.mean()*100:.1f}% | test: {y_test.mean()*100:.1f}%")

    preprocessor = make_preprocessor(features_genre)
    modelos      = definir_modelos(preprocessor)
    cv           = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    results_cv  = {}
    results_test = {}

    for nombre, pipe in modelos.items():
        print(f"\n[train] Entrenando: {nombre} ...")
        t0 = time.time()

        # Validacion cruzada (5-fold, agrupado por cancion)
        cv_scores = cross_validate(
            pipe, X_train, y_train, groups=g_train, cv=cv,
            scoring={"auc": "roc_auc", "f1": "f1", "recall": "recall", "precision": "precision"},
            return_train_score=True, n_jobs=1
        )
        results_cv[nombre] = cv_scores

        # Entrenamiento final sobre todo el train
        pipe.fit(X_train, y_train)

        # Evaluacion en test
        metricas = evaluar_modelo(pipe, X_train, y_train, X_test, y_test, nombre)
        gap_cv   = cv_scores["train_auc"].mean() - cv_scores["test_auc"].mean()
        metricas["gap_auc_cv"] = gap_cv
        results_test[nombre]   = metricas

        elapsed = time.time() - t0
        print(f"  AUC test: {metricas['auc']:.4f} | F1: {metricas['f1']:.4f} | "
              f"Recall: {metricas['recall']:.4f} | Gap CV: {gap_cv:.4f} | {elapsed:.1f}s")

        # Guardar pipeline
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        nombre_archivo = nombre.lower().replace(" ", "_")
        joblib.dump(pipe, os.path.join(OUTPUT_DIR, f"{nombre_archivo}.joblib"))
        print(f"  Guardado: {OUTPUT_DIR}/{nombre_archivo}.joblib")

    # Tabla resumen
    print("\n" + "=" * 60)
    print("RESUMEN — Evaluacion en conjunto de prueba")
    print("=" * 60)
    tabla = tabla_comparativa(results_test)
    print(tabla.to_string())

    # Guardar tabla
    tabla.to_csv(os.path.join(OUTPUT_DIR, "resultados_modelos.csv"))

    # Mejor modelo por AUC
    mejor = max(results_test, key=lambda n: results_test[n]["auc"])
    print(f"\nMejor modelo (AUC-ROC): {mejor} "
          f"(AUC={results_test[mejor]['auc']:.4f}, "
          f"F1={results_test[mejor]['f1']:.4f}, "
          f"Recall={results_test[mejor]['recall']:.4f})")

    # Guardar mejor modelo tambien como best_model.joblib
    mejor_pipe = modelos[mejor]
    joblib.dump(mejor_pipe, os.path.join(OUTPUT_DIR, "best_model.joblib"))

    # KPIs
    print("\n" + "=" * 60)
    print("KPIs de negocio")
    print("=" * 60)
    kpis = verificar_kpis(results_test)
    print(kpis.pivot(index="Modelo", columns="KPI", values="Cumple").to_string())

    print("\n[train] Entrenamiento completado.")
    return {"results_test": results_test, "results_cv": results_cv, "mejor": mejor}


if __name__ == "__main__":
    run()



