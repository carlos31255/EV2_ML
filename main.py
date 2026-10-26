import os
import sys
import time
import joblib
import pandas as pd


def main():
    start = time.time()
    print("\n" + "=" * 60)
    print("  EV2 Machine Learning - Caso C: Spotify")
    print("  Pipeline completo: preprocesamiento -> modelos -> test -> graficos")
    print("=" * 60 + "\n")

    # 1. Pipeline (limpieza, deduplicacion, split, transformaciones)
    from src.pipeline import run as run_pipeline
    run_pipeline()

    # 2. Entrenamiento
    from src.train import run as run_train
    resultados_train = run_train()
    mejor_nombre = resultados_train["mejor"]

    # 3. Evaluacion final en test
    from src.test import run as run_test
    results = run_test()

    # 4. Graficos (no bloquea el pipeline si fallan)
    try:
        from src.graficos import run as run_graficos
        y_test = pd.read_csv(
            os.path.join("data", "processed", "test", "y_test.csv")
        ).squeeze()
        rf_path = os.path.join("output", "modelos", "random_forest.joblib")
        rf_pipe = joblib.load(rf_path) if os.path.exists(rf_path) else None
        run_graficos(results=results, y_test=y_test, rf_pipe=rf_pipe)
    except Exception as e:
        print(f"[main] ADVERTENCIA: no se generaron graficos ({e})")

    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print(f"  Pipeline completo finalizado en {elapsed:.1f}s")
    print(f"  Mejor modelo: {mejor_nombre}")
    print(f"  Modelos guardados en:   output/modelos/")
    print(f"  Tablas de resultados:   output/modelos/resultados/")
    print(f"  Datos procesados en:    data/processed/train/ y test/")
    print(f"  Graficos en:            output/graficos/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    sys.stdout.reconfigure(line_buffering=True)
    main()