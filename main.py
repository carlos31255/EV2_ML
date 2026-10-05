"""
main.py
-------
Punto de entrada del proyecto. Ejecuta el pipeline completo de punta a punta:

  1. pipeline.py  — Preprocesamiento y split en disco
  2. train.py     — Entrenamiento de modelos y validacion cruzada
  3. test.py      — Evaluacion final en conjunto de prueba
  4. graficos.py  — Generacion de todas las visualizaciones

Uso:
  conda activate ml_spotify
  python main.py

Requisitos:
  - data/processed/spotify_clean.csv  (generado por EV1)
  - environment.yml (conda) o requirements.txt (pip)
"""

import time
import pandas as pd
import os

def main():
    start = time.time()
    print("\n" + "=" * 60)
    print("  EV2 Machine Learning — Caso C: Spotify")
    print("  Pipeline completo: preprocesamiento -> modelos -> graficos")
    print("=" * 60 + "\n")

    # ── 1. Pipeline ──────────────────────────────────────────────────────────
    from src.pipeline import run as run_pipeline
    ctx = run_pipeline()
    train       = ctx["train"]
    test        = ctx["test"]

    # ── 2. Entrenamiento ─────────────────────────────────────────────────────
    from src.train import run as run_train
    resultados_train = run_train()
    results      = resultados_train["results_test"]
    mejor_nombre = resultados_train["mejor"]

    # ── 3. Evaluacion final en test ───────────────────────────────────────────
    from src.test import run as run_test
    run_test()

    # ── 4. Graficos ──────────────────────────────────────────────────────────
    import joblib
    from src.graficos import run as run_graficos

    df_raw   = pd.read_csv(os.path.join("data", "processed", "spotify_clean.csv"))
    y_test   = pd.read_csv(os.path.join("data", "processed", "test", "y_test.csv")).squeeze()

    # Cargar el pipeline de Random Forest para grafico de importancia
    rf_path = os.path.join("output", "modelos", "random_forest.joblib")
    rf_pipe = joblib.load(rf_path) if os.path.exists(rf_path) else None

    # Graficos: generacion de todas las visualizaciones EDA y modelacion
    run_graficos(df_raw=df_raw, results=results, y_test=y_test, rf_pipe=rf_pipe)

    # ── Resumen final ────────────────────────────────────────────────────────
    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print(f"  Pipeline completo finalizado en {elapsed:.1f}s")
    print(f"  Mejor modelo: {mejor_nombre}")
    print(f"  Modelos guardados en:   output/modelos/")
    print(f"  Graficos guardados en:  output/graficos/")
    print(f"  Datos procesados en:    data/processed/train/ y test/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    main()



