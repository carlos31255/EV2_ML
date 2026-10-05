"""
src/graficos.py
---------------
Genera todas las visualizaciones del proyecto y las guarda en output/graficos/.

Graficos producidos:
  EDA:
    01_distribucion_popularidad.png
    02_distribucion_audio_features.png
    03_heatmap_correlaciones.png
    04_outliers_boxplots.png
    05_popularidad_por_genero.png
  Modelos:
    06_curvas_roc.png
    07_matrices_confusion.png
    08_feature_importance_rf.png
    09_curva_ganancia_acumulada.png
  No supervisado:
    10_kmeans_seleccion_k.png
    11_kmeans_perfil_segmentos.png
    12_kmeans_segmentos_pca.png

Uso:
  python -m src.graficos
  from src.graficos import run, grafico_roc, grafico_importancia
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sin pantalla (subprocesos Windows)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import roc_curve, ConfusionMatrixDisplay

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.family":       "sans-serif",
    "axes.facecolor":    "#1a1a2e",
    "figure.facecolor":  "#0f0f1a",
    "text.color":        "#e0e0e0",
    "axes.labelcolor":   "#e0e0e0",
    "xtick.color":       "#e0e0e0",
    "ytick.color":       "#e0e0e0",
    "axes.edgecolor":    "#3a3a5c",
    "grid.color":        "#2a2a4a",
    "grid.alpha":        0.4,
})

SPOTIFY_GREEN = "#1DB954"
PALETTE       = [SPOTIFY_GREEN, "#1E90FF", "#FFD700", "#FF6B6B", "#B19CD9", "#FF8C00"]
OUTPUT_DIR    = os.path.join("output", "graficos")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save(fig, nombre: str) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ruta = os.path.join(OUTPUT_DIR, nombre)
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[graficos] Guardado: {ruta}")


# ── EDA ───────────────────────────────────────────────────────────────────────

def grafico_distribucion_popularidad(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["popularity"], bins=50, color=SPOTIFY_GREEN, alpha=0.8, edgecolor="none")
    ax.set_xlabel("Popularidad (0-100)")
    ax.set_ylabel("Canciones")
    ax.set_title("Distribucion de popularidad — Spotify Tracks Dataset", fontsize=13, pad=12)
    ax.axvline(df["popularity"].quantile(0.80), color="#FFD700", linestyle="--",
               linewidth=1.5, label="P80 (umbral hit)")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    _save(fig, "01_distribucion_popularidad.png")


def grafico_audio_features(df: pd.DataFrame) -> None:
    features = ["danceability", "energy", "valence", "acousticness", "tempo", "loudness"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, feat in zip(axes.flat, features):
        ax.hist(df[feat], bins=40, color=SPOTIFY_GREEN, alpha=0.8, edgecolor="none")
        ax.set_title(feat, fontsize=10)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.suptitle("Distribucion de audio features", fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, "02_distribucion_audio_features.png")


def grafico_correlaciones(df: pd.DataFrame, features: list) -> None:
    corr = df[features + ["popularity"]].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                linewidths=0.5, linecolor="#0f0f1a", ax=ax,
                annot_kws={"size": 7})
    ax.set_title("Heatmap de correlaciones (audio features + popularidad)", fontsize=12, pad=12)
    fig.tight_layout()
    _save(fig, "03_heatmap_correlaciones.png")


def grafico_outliers(df: pd.DataFrame, features: list) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    data    = [df[f].dropna() for f in features]
    bp = ax.boxplot(data, patch_artist=True, medianprops={"color": SPOTIFY_GREEN})
    for patch, color in zip(bp["boxes"], PALETTE * 3):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(features)+1))
    ax.set_xticklabels(features, rotation=35, ha="right", fontsize=9)
    ax.set_title("Outliers — audio features continuas", fontsize=12, pad=12)
    _save(fig, "04_outliers_boxplots.png")


def grafico_popularidad_genero(df: pd.DataFrame, top_n: int = 20) -> None:
    pop_genero = df.groupby("track_genre")["popularity"].mean().sort_values(ascending=False)
    top    = pop_genero.head(top_n)
    bottom = pop_genero.tail(top_n)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    top.plot(kind="barh", ax=ax1, color=SPOTIFY_GREEN, alpha=0.85)
    ax1.set_title(f"Top {top_n} generos — mayor popularidad promedio", fontsize=11)
    ax1.set_xlabel("Popularidad promedio")
    ax1.invert_yaxis()
    bottom.sort_values().plot(kind="barh", ax=ax2, color="#FF6B6B", alpha=0.85)
    ax2.set_title(f"Bottom {top_n} generos — menor popularidad promedio", fontsize=11)
    ax2.set_xlabel("Popularidad promedio")
    ax2.invert_yaxis()
    fig.suptitle("Popularidad promedio por genero musical", fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, "05_popularidad_por_genero.png")


# ── Modelos ───────────────────────────────────────────────────────────────────

def grafico_roc(results: dict, y_test) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for (nombre, r), color in zip(results.items(), PALETTE):
        fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{nombre} (AUC={r['auc']:.4f})")
    ax.plot([0, 1], [0, 1], "w--", linewidth=1, alpha=0.5, label="Aleatorio (AUC=0.50)")
    ax.set_xlabel("Tasa de falsos positivos (FPR)")
    ax.set_ylabel("Tasa de verdaderos positivos (TPR)")
    ax.set_title("Curvas ROC — comparativa de modelos", fontsize=12, pad=12)
    ax.legend(fontsize=9)
    _save(fig, "06_curvas_roc.png")


def grafico_matrices_confusion(results: dict, y_test) -> None:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1:
        axes = [axes]
    for ax, (nombre, r) in zip(axes, results.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test, r["y_pred"], ax=ax,
            display_labels=["No-hit", "Hit"],
            colorbar=False, cmap="Greens"
        )
        ax.set_title(nombre, fontsize=10)
    fig.suptitle("Matrices de confusion — conjunto de prueba", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, "07_matrices_confusion.png")


def grafico_importancia_rf(pipe, feature_names: list, top_n: int = 20) -> None:
    """Importancia de variables agrupada por variable original (Random Forest)."""
    rf = pipe.named_steps["clf"]
    importancias = rf.feature_importances_

    # Reconstruir nombres de features tras OHE
    ohe_cols = list(pipe.named_steps["pre"].named_transformers_["ohe"]
                    .get_feature_names_out(["key", "time_signature"]))
    scale_cols  = list(pipe.named_steps["pre"].transformers_[0][2])
    pass_cols   = list(pipe.named_steps["pre"].transformers_[2][2])
    all_cols    = scale_cols + ohe_cols + pass_cols

    df_imp = pd.DataFrame({"feature": all_cols, "importancia": importancias})

    # Agrupar genre_* en "Genero (multi-hot)"
    def grupo(f):
        if f.startswith("genre_"): return "Genero (multi-hot)"
        if f.startswith("key_"):   return "key (OHE)"
        if f.startswith("time_"):  return "time_signature (OHE)"
        return f
    df_imp["grupo"] = df_imp["feature"].apply(grupo)
    df_grp = df_imp.groupby("grupo")["importancia"].sum().sort_values(ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(df_grp.index, df_grp.values, color=SPOTIFY_GREEN, alpha=0.85)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8, color="#e0e0e0")
    ax.set_title("Importancia de variables — Random Forest (agrupada)", fontsize=12, pad=12)
    ax.set_xlabel("Importancia relativa")
    fig.tight_layout()
    _save(fig, "08_feature_importance_rf.png")


def grafico_ganancia_acumulada(results: dict, y_test) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    n      = len(y_test)
    pct    = np.arange(1, n+1) / n * 100
    base   = np.cumsum(np.sort(y_test)[::-1]) / y_test.sum() * 100
    ax.plot(pct, base, "w--", linewidth=1, alpha=0.4, label="Modelo perfecto")
    ax.plot([0, 100], [0, 100], "w:", linewidth=1, alpha=0.3, label="Aleatorio")
    for (nombre, r), color in zip(results.items(), PALETTE):
        idx  = np.argsort(r["y_proba"])[::-1]
        gain = np.cumsum(np.array(y_test)[idx]) / y_test.sum() * 100
        ax.plot(pct, gain, color=color, linewidth=2,
                label=f"{nombre} (Lift@10%={r['lift_top10']:.2f}x)")
    ax.axvline(10, color="#FFD700", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("% de canciones evaluadas (ordenadas por probabilidad)")
    ax.set_ylabel("% de hits capturados")
    ax.set_title("Curva de ganancia acumulada — top-10% marcado", fontsize=12, pad=12)
    ax.legend(fontsize=9)
    _save(fig, "09_curva_ganancia_acumulada.png")


# ── Funcion principal ─────────────────────────────────────────────────────────

def run(df_raw: pd.DataFrame = None, results: dict = None,
        y_test=None, rf_pipe=None, features_scale: list = None) -> None:
    """
    Genera todos los graficos. Los argumentos se pasan desde main.py.
    Si no se pasan datos, carga desde disco.
    """
    if df_raw is None:
        df_raw = pd.read_csv(os.path.join("data", "processed", "spotify_clean.csv"))
    if features_scale is None:
        from src.pipeline import FEATURES_SCALE as fs
        features_scale = fs

    print("=" * 60)
    print("GRAFICOS — Generando visualizaciones")
    print("=" * 60)

    # EDA
    grafico_distribucion_popularidad(df_raw)
    grafico_audio_features(df_raw)
    grafico_correlaciones(df_raw, features_scale)
    grafico_outliers(df_raw, features_scale)
    grafico_popularidad_genero(df_raw)

    # Modelos (solo si se pasan resultados)
    if results is not None and y_test is not None:
        grafico_roc(results, y_test)
        grafico_matrices_confusion(results, y_test)
        grafico_ganancia_acumulada(results, y_test)
        if rf_pipe is not None:
            feature_names = pd.read_csv(
                os.path.join("data", "processed", "feature_names.csv")
            )["feature"].tolist()
            grafico_importancia_rf(rf_pipe, feature_names)

    print("\n[graficos] Visualizaciones completadas.")


if __name__ == "__main__":
    run()

