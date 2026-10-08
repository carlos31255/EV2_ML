"""
src/metricas.py
---------------
Funciones de evaluacion de modelos: AUC, PR-AUC, F1, Recall, Precision,
Accuracy, Lift acumulado @ top-10%, gap de sobreajuste.

Uso:
  from src.metricas import evaluar_modelo, tabla_comparativa
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score,
    precision_score, accuracy_score, average_precision_score,
    classification_report
)


def evaluar_modelo(pipe, X_train, y_train, X_test, y_test, nombre: str) -> dict:
    """
    Calcula las metricas principales de un pipeline entrenado.

    Retorna un dict con:
        accuracy, precision, recall, f1, auc (test), pr_auc,
        auc_train, gap_auc, lift_top10, y_proba, y_pred
    """
    proba_test  = pipe.predict_proba(X_test)[:, 1]
    proba_train = pipe.predict_proba(X_train)[:, 1]
    pred_test   = (proba_test >= 0.5).astype(int)

    auc_test  = roc_auc_score(y_test,  proba_test)
    auc_train = roc_auc_score(y_train, proba_train)
    gap_auc   = auc_train - auc_test

    # Lift acumulado al top-10%: cuantos hits hay en el 10% con mayor probabilidad
    n_top    = max(1, int(len(y_test) * 0.10))
    top_idx  = np.argsort(proba_test)[::-1][:n_top]
    hit_rate_base = y_test.mean()
    hit_rate_top  = np.array(y_test)[top_idx].mean()
    lift_top10    = hit_rate_top / hit_rate_base if hit_rate_base > 0 else 0.0

    return {
        "nombre":    nombre,
        "accuracy":  accuracy_score(y_test, pred_test),
        "precision": precision_score(y_test, pred_test, zero_division=0),
        "recall":    recall_score(y_test, pred_test, zero_division=0),
        "f1":        f1_score(y_test, pred_test, zero_division=0),
        "auc":       auc_test,
        "auc_train": auc_train,
        "pr_auc":    average_precision_score(y_test, proba_test),
        "gap_auc":   gap_auc,
        "lift_top10": lift_top10,
        "y_proba":   proba_test,
        "y_pred":    pred_test,
    }


def tabla_comparativa(results: dict) -> pd.DataFrame:
    """
    Construye una tabla comparativa de metricas para todos los modelos.

    Args:
        results: dict {nombre_modelo: dict_metricas}
    Returns:
        DataFrame con columnas: Modelo, Accuracy, Precision, Recall, F1, AUC-ROC,
                                PR-AUC, AUC train, Gap AUC, Lift@10%, Sobreajuste
    """
    filas = []
    for nombre, r in results.items():
        filas.append({
            "Modelo":      nombre,
            "Accuracy":    round(r["accuracy"],  4),
            "Precision":   round(r["precision"], 4),
            "Recall":      round(r["recall"],    4),
            "F1-Score":    round(r["f1"],        4),
            "AUC-ROC":     round(r["auc"],       4),
            "PR-AUC":      round(r["pr_auc"],    4),
            "AUC train":   round(r["auc_train"], 4),
            "Gap AUC":     round(r["gap_auc"],   4),
            "Lift @10%":   round(r["lift_top10"],2),
            "Sobreajuste": "Si" if r["gap_auc"] >= 0.05 else "No",
        })
    return pd.DataFrame(filas).set_index("Modelo")


def verificar_kpis(results: dict) -> pd.DataFrame:
    """
    Compara los resultados de cada modelo contra los KPIs de negocio definidos:
        - AUC-ROC >= 0.80
        - Recall   >= 0.75  (detectar al menos 75% de hits reales)
        - Lift @10%>= 2.0   (priorizar 2x mejor que al azar)
        - Gap AUC  <  0.05  (sin sobreajuste significativo)

    Retorna DataFrame con columna "KPI cumplido" (Si/No) por modelo y metrica.
    """
    KPI_TARGETS = {
        "AUC-ROC":    ("auc",       ">=", 0.80),
        "Recall":     ("recall",    ">=", 0.75),
        "Lift @10%":  ("lift_top10",">=", 2.00),
        "Gap AUC":    ("gap_auc",   "<",  0.05),
    }

    filas = []
    for nombre, r in results.items():
        for kpi_name, (key, op, meta) in KPI_TARGETS.items():
            valor = r[key]
            if op == ">=":
                cumple = valor >= meta
            else:
                cumple = valor < meta
            filas.append({
                "Modelo": nombre, "KPI": kpi_name,
                "Meta": f"{op} {meta}", "Valor": round(valor, 4),
                "Cumple": "Si" if cumple else "No",
            })
    return pd.DataFrame(filas)