"""
src/metricas.py
---------------
Funciones de evaluacion de modelos: AUC, PR-AUC, F1, Recall, Precision,
Accuracy, Lift acumulado @ top-10%, gap de sobreajuste, y la politica de
umbral del modelo oficial (umbral por genero + numero de generos) con sus
metricas de equidad.

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


def evaluar_modelo(pipe, X_train, y_train, X_test, y_test, nombre: str, umbral=0.5) -> dict:
    """
    Calcula las metricas principales de un pipeline entrenado.

    Retorna un dict con:
        accuracy, precision, recall, f1, auc (test), pr_auc,
        auc_train, gap_auc, lift_top10, y_proba, y_pred

    umbral puede ser un numero (0.5 por defecto) o un arreglo con un umbral por cancion
    (politica de umbral, mas abajo en este archivo). El umbral no cambia AUC, PR-AUC, gap ni lift.
    """
    proba_test  = pipe.predict_proba(X_test)[:, 1]
    proba_train = pipe.predict_proba(X_train)[:, 1]
    pred_test   = (proba_test >= np.asarray(umbral)).astype(int)

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


# ============================================================================
# POLITICA DE UMBRAL DEL MODELO OFICIAL
# ============================================================================
# Reparte el recall entre generos y lo iguala entre canciones de 1, 2 y 3+ generos SIN cambiar
# el ranking del modelo (AUC y lift no cambian). Reduce el sesgo de genero, no lo elimina.
#
# 1. Umbral por genero: para cada genero, el umbral que logra recall R_OBJ (0.78) sobre
#    sus hits, calculado con probabilidades FUERA DE MUESTRA de train (CV agrupada).
#    Si el genero tiene pocos hits, su umbral se encoge hacia el umbral global:
#        u_g = a * u_genero + (1 - a) * u_global,   a = hits_g / (hits_g + K_ENCOGIMIENTO)
#    Una cancion con varios generos usa el promedio de los umbrales de sus generos.
# 2. Correccion por numero de generos (1, 2, 3+): desplazamiento aditivo para que el
#    recall fuera de muestra de cada grupo sea R_OBJ.
#
# R_OBJ = 0.78 (margen sobre el KPI de 0.75) y K_ENCOGIMIENTO = 30 son decisiones de diseno,
# sin analisis de sensibilidad. Los umbrales se calibran solo con train (test no se usa aqui).
#
# Uso:
#   from src.metricas import ajustar_politica, aplicar_politica, umbral_politica

R_OBJ = 0.78
K_ENCOGIMIENTO = 30
GRUPOS_N = (1, 2, 3)          # 3 = "3 o mas generos"


def grupo_n_generos(G) -> np.ndarray:
    """1, 2 o 3 (= 3 o mas) segun cuantas columnas genre_* tiene activas cada cancion."""
    return np.minimum(np.asarray(G, dtype=float).sum(axis=1), 3).astype(int)


def _umbral_cancion(G, u_genero) -> np.ndarray:
    G = np.asarray(G, dtype=float)
    return (G @ u_genero) / np.maximum(G.sum(axis=1), 1)


def ajustar_politica(prob_oof, y, G, r_obj: float = R_OBJ, k: float = K_ENCOGIMIENTO) -> dict:
    """Calibra los umbrales con probabilidades fuera de muestra de train."""
    prob_oof, y = np.asarray(prob_oof, dtype=float), np.asarray(y).astype(int)
    Gb = np.asarray(G).astype(bool)

    u_global = float(np.quantile(prob_oof[y == 1], 1 - r_obj))
    u_genero = np.full(Gb.shape[1], u_global)
    for j in range(Gb.shape[1]):
        pos = prob_oof[Gb[:, j] & (y == 1)]
        if len(pos):
            a = len(pos) / (len(pos) + k)
            u_genero[j] = a * np.quantile(pos, 1 - r_obj) + (1 - a) * u_global

    margen = prob_oof - _umbral_cancion(Gb, u_genero)
    n = grupo_n_generos(Gb)
    ajuste_n = {int(g): float(np.quantile(margen[(n == g) & (y == 1)], 1 - r_obj)) for g in GRUPOS_N}
    return {"r_obj": r_obj, "k": k, "u_global": u_global,
            "u_genero": u_genero, "ajuste_n": ajuste_n}


def umbral_politica(G, politica: dict) -> np.ndarray:
    """Umbral de decision de cada cancion segun sus generos."""
    n = grupo_n_generos(G)
    ajuste = np.array([politica["ajuste_n"][int(g)] for g in n])
    return _umbral_cancion(G, politica["u_genero"]) + ajuste


def aplicar_politica(prob, G, politica: dict) -> np.ndarray:
    """Prediccion 0/1 con la politica de umbral."""
    return (np.asarray(prob) >= umbral_politica(G, politica)).astype(int)


def resumen_equidad(y, pred, G) -> dict:
    """Metricas de equidad entre generos y por numero de generos."""
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    Gf = np.asarray(G, dtype=float)
    hits, tp = Gf.T @ y, Gf.T @ (y * pred)
    ok = hits > 0
    rec_g = tp[ok] / hits[ok]
    hr_g = (hits / np.maximum(Gf.sum(axis=0), 1))[ok]
    bajo = hr_g < 0.10
    out = {"recall_macro_generos": rec_g.mean(),
           "recall_generos_hit_rate_bajo_10pct": tp[ok][bajo].sum() / hits[ok][bajo].sum(),
           "pct_generos_recall_kpi": (rec_g >= 0.75).mean() * 100}
    n = grupo_n_generos(Gf)
    for g, et in zip(GRUPOS_N, ("1", "2", "3+")):
        m = n == g
        out[f"marcadas_sobre_hit_rate_{et}"] = pred[m].mean() / y[m].mean()
        out[f"recall_{et}_generos"] = (y[m] & pred[m]).sum() / y[m].sum()
        out[f"precision_{et}_generos"] = (y[m] & pred[m]).sum() / max(pred[m].sum(), 1)
    return out
