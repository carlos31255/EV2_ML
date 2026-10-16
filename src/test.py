"""
src/test.py
-----------
Evaluacion final de los modelos entrenados sobre data/processed/test.
Aqui recae toda la evaluacion final: AUC, precision, recall, F1, lift,
PR-AUC y verificacion de los 4 KPIs de negocio.
Los tres modelos se comparan con umbral 0.5. El modelo OFICIAL es el elegido por CV
con la politica de umbral por genero y numero de generos (src/metricas.py), calibrada
en train; sus KPIs son los que cuentan.
El modelo final es el elegido por AUC de CV en src/train.py (resultados_cv.csv);
el test solo se usa para reportar.

Uso:
  python -m src.test
  from src.test import run
"""

import os
import joblib
import pandas as pd
from src.metricas import (evaluar_modelo, tabla_comparativa, verificar_kpis,
                          umbral_politica, resumen_equidad)

TEST_DIR   = os.path.join("data", "processed", "test")
TRAIN_DIR  = os.path.join("data", "processed", "train")
OUTPUT_DIR = os.path.join("output", "modelos")

ARCHIVOS_MODELOS = {
    "Regresion Logistica":  "regresion_logistica.joblib",
    "Random Forest":        "random_forest.joblib",
    "HistGradientBoosting": "histgradientboosting.joblib",
}


def _hay_modelos() -> bool:
    """True si existe al menos un .joblib de los modelos esperados."""
    return any(
        os.path.exists(os.path.join(OUTPUT_DIR, f))
        for f in ARCHIVOS_MODELOS.values()
    )


def _mejor_por_cv(results: dict):
    """Nombre del modelo con mayor AUC de CV (output/modelos/resultados_cv.csv), o None."""
    ruta = os.path.join(OUTPUT_DIR, "resultados_cv.csv")
    if not os.path.exists(ruta):
        return None
    cv = pd.read_csv(ruta, index_col=0)
    for nombre in cv["auc_cv"].sort_values(ascending=False).index:
        if nombre in results:
            return nombre
    return None


def run() -> dict:
    """Carga modelos serializados, evalua en test y guarda reporte."""
    print("=" * 60)
    print("TEST — Evaluacion final en conjunto de prueba")
    print("=" * 60)

    # Chequeo: sin modelos no se puede evaluar
    if not _hay_modelos():
        print(f"[test] No se encontro ningun modelo .joblib en {OUTPUT_DIR}.")
        print("[test] Ejecuta 'python main.py' o 'python -m src.train' primero.")
        return {}

    X_train = pd.read_csv(os.path.join(TRAIN_DIR, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(TRAIN_DIR, "y_train.csv")).squeeze()
    X_test  = pd.read_csv(os.path.join(TEST_DIR,  "X_test.csv"))
    y_test  = pd.read_csv(os.path.join(TEST_DIR,  "y_test.csv")).squeeze()

    results = {}
    for nombre, archivo in ARCHIVOS_MODELOS.items():
        ruta = os.path.join(OUTPUT_DIR, archivo)
        if not os.path.exists(ruta):
            print(f"[test] ADVERTENCIA: no se encontro {ruta}, saltando.")
            continue
        pipe = joblib.load(ruta)
        metricas = evaluar_modelo(pipe, X_train, y_train, X_test, y_test, nombre)
        results[nombre] = metricas
        print(f"[test] {nombre}: AUC={metricas['auc']:.4f} | "
              f"PR-AUC={metricas['pr_auc']:.4f} | Precision={metricas['precision']:.4f} | "
              f"Recall={metricas['recall']:.4f} | Lift@10%={metricas['lift_top10']:.2f}")

    # Modelo oficial: el elegido por CV + politica de umbral calibrada en train
    mejor = _mejor_por_cv(results)
    ruta_pol = os.path.join(OUTPUT_DIR, "politica_umbral.joblib")
    if mejor is not None and os.path.exists(ruta_pol):
        politica = joblib.load(ruta_pol)
        G_test = X_test[politica["features_genre"]].values
        nombre_of = f"{mejor} + umbral por genero (oficial)"
        pipe = joblib.load(os.path.join(OUTPUT_DIR, ARCHIVOS_MODELOS[mejor]))
        metricas = evaluar_modelo(pipe, X_train, y_train, X_test, y_test, nombre_of,
                                  umbral=umbral_politica(G_test, politica))
        results[nombre_of] = metricas
        print(f"[test] {nombre_of}: Precision={metricas['precision']:.4f} | Recall={metricas['recall']:.4f}")

        # Equidad entre generos y por numero de generos: umbral 0.5 vs politica oficial
        equidad = pd.DataFrame({
            f"{mejor} (umbral 0.5)": resumen_equidad(y_test, results[mejor]["y_pred"], G_test),
            nombre_of:               resumen_equidad(y_test, metricas["y_pred"], G_test)})
        equidad.round(4).to_csv(os.path.join(OUTPUT_DIR, "equidad_test.csv"))
        print("\nEquidad (test):")
        print(equidad.round(3).to_string())
    else:
        print("[test] ADVERTENCIA: no hay politica_umbral.joblib; se reporta solo umbral 0.5.")

    # Tabla comparativa
    tabla = tabla_comparativa(results)
    print("\n" + tabla.to_string())
    tabla.to_csv(os.path.join(OUTPUT_DIR, "resultados_test.csv"))

    # KPIs de negocio (los 4)
    kpis = verificar_kpis(results)
    kpis.to_csv(os.path.join(OUTPUT_DIR, "kpis_cumplidos.csv"), index=False)
    print("\nKPIs de negocio:")
    print(kpis.pivot(index="Modelo", columns="KPI", values="Cumple").to_string())

    # Detalle del modelo final: el elegido por AUC de CV en train.py (no por test), con su politica
    if mejor is None:
        print("\n[test] ADVERTENCIA: no hay resultados_cv.csv; no se puede identificar "
              "el modelo elegido por CV. Ejecuta 'python -m src.train' primero.")
    else:
        print(f"\n[test] Modelo elegido por AUC de CV: {mejor}")
        mejor_test = max((n for n in ARCHIVOS_MODELOS if n in results), key=lambda n: results[n]["auc"])
        if mejor_test != mejor:
            print(f"[test] Nota: en test el mayor AUC lo tiene {mejor_test}; "
                  f"la seleccion no se cambia por el resultado de test.")
        oficial = next((n for n in results if n.endswith("(oficial)")), mejor)
        detalle = {k: v for k, v in results[oficial].items()
                   if isinstance(v, (int, float))}
        for k, v in detalle.items():
            print(f"   {k}: {v:.4f}")
        pd.Series(detalle, name=oficial).to_csv(
            os.path.join(OUTPUT_DIR, "mejor_modelo_test.csv")
        )

    print("\n[test] Evaluacion completada.")
    return results


if __name__ == "__main__":
    run()