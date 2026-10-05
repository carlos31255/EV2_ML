"""
src/test.py
-----------
Evalua los modelos entrenados sobre el conjunto de prueba
y genera el reporte de metricas en output/modelos/.

Uso:
  python -m src.test
  from src.test import run
"""

import os
import joblib
import pandas as pd
import numpy as np
from src.metricas import evaluar_modelo, tabla_comparativa, verificar_kpis

TEST_DIR   = os.path.join("data", "processed", "test")
TRAIN_DIR  = os.path.join("data", "processed", "train")
OUTPUT_DIR = os.path.join("output", "modelos")


def run() -> dict:
    """Carga modelos serializados, evalua en test y guarda reporte."""
    print("=" * 60)
    print("TEST — Evaluacion final en conjunto de prueba")
    print("=" * 60)

    X_train = pd.read_csv(os.path.join(TRAIN_DIR, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(TRAIN_DIR, "y_train.csv")).squeeze()
    X_test  = pd.read_csv(os.path.join(TEST_DIR,  "X_test.csv"))
    y_test  = pd.read_csv(os.path.join(TEST_DIR,  "y_test.csv")).squeeze()

    # Cargar modelos
    archivos_modelos = {
        "Regresion Logistica":  "regresion_logistica.joblib",
        "Random Forest":        "random_forest.joblib",
        "HistGradientBoosting": "histgradientboosting.joblib",
    }

    results = {}
    for nombre, archivo in archivos_modelos.items():
        ruta = os.path.join(OUTPUT_DIR, archivo)
        if not os.path.exists(ruta):
            print(f"[test] ADVERTENCIA: no se encontro {ruta}, saltando.")
            continue
        pipe = joblib.load(ruta)
        metricas = evaluar_modelo(pipe, X_train, y_train, X_test, y_test, nombre)
        results[nombre] = metricas
        print(f"[test] {nombre}: AUC={metricas['auc']:.4f} | "
              f"F1={metricas['f1']:.4f} | Recall={metricas['recall']:.4f} | "
              f"Lift@10%={metricas['lift_top10']:.2f}")

    # Tabla resumen
    tabla = tabla_comparativa(results)
    print("\n" + tabla.to_string())
    tabla.to_csv(os.path.join(OUTPUT_DIR, "resultados_test.csv"))

    # KPIs
    kpis = verificar_kpis(results)
    kpis.to_csv(os.path.join(OUTPUT_DIR, "kpis_cumplidos.csv"), index=False)
    print("\nKPIs de negocio:")
    print(kpis.pivot(index="Modelo", columns="KPI", values="Cumple").to_string())

    print("\n[test] Evaluacion completada.")
    return results


if __name__ == "__main__":
    run()
