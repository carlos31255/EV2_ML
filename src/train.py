"""
src/train.py
------------
Entrena los modelos supervisados con validacion cruzada agrupada por cancion
y los guarda en output/modelos/. NO evalua en test (eso es src/test.py).
El mejor modelo se elige por AUC de validacion cruzada, no por test.
Ademas mide el aporte del genero (ablacion: mejor modelo con y sin genero, por CV).

Uso:
  python -m src.train
"""

import os
import time
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_validate

from src.pipeline import FEATURES_SCALE, FEATURES_CAT, FEATURES_BIN

TRAIN_DIR  = os.path.join("data", "processed", "train")
OUTPUT_DIR = os.path.join("output", "modelos")
SEED       = 42
CV_FOLDS   = 5


def make_preprocessor(features_genre) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), FEATURES_SCALE),
            ("ohe",   OneHotEncoder(handle_unknown="ignore", sparse_output=False), FEATURES_CAT),
            ("pass",  "passthrough", FEATURES_BIN + features_genre),
        ],
        remainder="drop"
    )


def definir_modelos(preprocessor: ColumnTransformer) -> dict:
    return {
        "Regresion Logistica": Pipeline([
            ("pre", preprocessor),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                       random_state=SEED, solver="saga", n_jobs=1))
        ]),
        "Random Forest": Pipeline([
            ("pre", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=15,
                                           min_samples_leaf=10, class_weight="balanced",
                                           random_state=SEED, n_jobs=1))
        ]),
        "HistGradientBoosting": Pipeline([
            ("pre", preprocessor),
            ("clf", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                   max_depth=6, class_weight="balanced",
                                                   random_state=SEED))
        ]),
    }


def ablacion_genero(nombre, X_train, y_train, g_train, cv, auc_con_genero) -> dict:
    """
    Mismo modelo sin las columnas genre_* (se descartan con remainder='drop').
    Devuelve el AUC de CV con y sin genero; no usa el conjunto de test.
    """
    pipe_sin = definir_modelos(make_preprocessor([]))[nombre]
    sc = cross_validate(pipe_sin, X_train, y_train, groups=g_train, cv=cv,
                        scoring={"auc": "roc_auc"}, n_jobs=1)
    return {
        "modelo":         nombre,
        "auc_con_genero": float(auc_con_genero),
        "auc_sin_genero": float(sc["test_auc"].mean()),
    }


def run() -> dict:
    print("=" * 60)
    print("TRAIN - Entrenamiento de modelos supervisados")
    print("=" * 60)

    feat_names = pd.read_csv(os.path.join("data", "processed", "feature_names.csv"))
    features_genre = [f for f in feat_names["feature"] if f.startswith("genre_")]

    X_train = pd.read_csv(os.path.join(TRAIN_DIR, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(TRAIN_DIR, "y_train.csv")).squeeze()
    g_train = pd.read_csv(os.path.join(TRAIN_DIR, "g_train.csv")).squeeze()

    print(f"[train] X_train: {X_train.shape} | Hit rate train: {y_train.mean()*100:.1f}%")

    preprocessor = make_preprocessor(features_genre)
    modelos      = definir_modelos(preprocessor)
    cv           = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    results_cv = {}
    resumen    = {}
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for nombre, pipe in modelos.items():
        print(f"\n[train] Entrenando: {nombre} ...")
        t0 = time.time()

        cv_scores = cross_validate(
            pipe, X_train, y_train, groups=g_train, cv=cv,
            scoring={"auc": "roc_auc", "f1": "f1", "recall": "recall", "precision": "precision"},
            return_train_score=True, n_jobs=1
        )
        results_cv[nombre] = cv_scores

        pipe.fit(X_train, y_train)

        resumen[nombre] = {
            "auc_cv":       cv_scores["test_auc"].mean(),
            "auc_train_cv": cv_scores["train_auc"].mean(),
            "gap_auc_cv":   cv_scores["train_auc"].mean() - cv_scores["test_auc"].mean(),
            "f1_cv":        cv_scores["test_f1"].mean(),
            "recall_cv":    cv_scores["test_recall"].mean(),
            "precision_cv": cv_scores["test_precision"].mean(),
        }
        r = resumen[nombre]
        print(f"  AUC CV: {r['auc_cv']:.4f} | Gap CV: {r['gap_auc_cv']:.4f} | "
              f"{time.time() - t0:.1f}s")

        nombre_archivo = nombre.lower().replace(" ", "_")
        joblib.dump(pipe, os.path.join(OUTPUT_DIR, f"{nombre_archivo}.joblib"))

    tabla_cv = pd.DataFrame(resumen).T
    print("\n" + "=" * 60)
    print(tabla_cv.round(4).to_string())
    tabla_cv.to_csv(os.path.join(OUTPUT_DIR, "resultados_cv.csv"))

    mejor = tabla_cv["auc_cv"].idxmax()
    joblib.dump(modelos[mejor], os.path.join(OUTPUT_DIR, "best_model.joblib"))
    print(f"\n[train] Mejor modelo por AUC en CV: {mejor}")

    # Ablacion: peso del genero en el mejor modelo (AUC de CV con y sin genero)
    print(f"[train] Ablacion de genero sobre {mejor} ...")
    abl = ablacion_genero(mejor, X_train, y_train, g_train, cv, resumen[mejor]["auc_cv"])
    print(f"[train] AUC CV con genero: {abl['auc_con_genero']:.4f} | "
          f"sin genero: {abl['auc_sin_genero']:.4f} | "
          f"aporte: {abl['auc_con_genero'] - abl['auc_sin_genero']:+.4f}")
    pd.Series(abl).to_csv(os.path.join(OUTPUT_DIR, "ablacion_genero.csv"), header=["valor"])

    return {"results_cv": results_cv, "tabla_cv": tabla_cv, "mejor": mejor,
            "ablacion_genero": abl}


if __name__ == "__main__":
    run()
