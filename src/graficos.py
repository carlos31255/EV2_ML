"""
src/graficos.py
---------------
Todas las visualizaciones del proyecto. Los notebooks importan de aqui.
Guarda en output/graficos/.

Exploracion inicial (0_exploracion_inicial.ipynb):
  00_generos_por_cancion.png
EDA (sobre data/raw/train con transformaciones aplicadas):
  01_distribucion_popularidad.png
  02_distribuciones_numericas.png
  03_hit_rate_por_genero.png
  04_hit_rate_n_generos.png
  05_correlaciones.png
  06_features_por_clase.png
  07_popularidad_cero.png
Modelos:
  08_curvas_roc.png
  09_curvas_pr.png
  10_matrices_confusion.png
  11_feature_importance_rf.png
  12_curva_ganancia_acumulada.png
Analisis de modelos (2_Modelacion.ipynb):
  13_recall_por_genero.png
  14_importancia_permutacion.png
  15_kmeans_silhouette.png
  (16 a 19 los dibuja el notebook: recall vs hit rate del genero y K-Means ampliado)
  20_politica_umbral.png

Uso:
  python -m src.graficos
  from src import graficos as gr
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib

# En scripts: backend sin pantalla. En Jupyter: se deja el backend inline.
EN_NOTEBOOK = "ipykernel" in sys.modules
if not EN_NOTEBOOK:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import (roc_curve, precision_recall_curve, roc_auc_score,
                             average_precision_score, ConfusionMatrixDisplay,
                             silhouette_score)

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.family":       "sans-serif",
    "axes.facecolor":    "#ffffff",
    "figure.facecolor":  "#f8f9fa",
    "text.color":        "#212529",
    "axes.labelcolor":   "#212529",
    "xtick.color":       "#212529",
    "ytick.color":       "#212529",
    "axes.edgecolor":    "#dee2e6",
    "grid.color":        "#e9ecef",
    "grid.alpha":        0.8,
})

SPOTIFY_GREEN = "#1DB954"
RED           = "#FF6B6B"
PALETTE       = [SPOTIFY_GREEN, "#1E90FF", "#FFD700", RED, "#B19CD9", "#FF8C00"]
OUTPUT_DIR    = os.path.join("output", "graficos")

NUM_COLS  = ["danceability", "energy", "loudness", "valence", "tempo",
             "acousticness", "speechiness", "instrumentalness", "liveness", "duration_ms"]
CORR_COLS = NUM_COLS + ["explicit", "mode", "key", "time_signature"]
VARS_BOX  = ["danceability", "energy", "loudness", "valence",
             "acousticness", "instrumentalness"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save(fig, nombre: str) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ruta = os.path.join(OUTPUT_DIR, nombre)
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    if EN_NOTEBOOK:
        plt.show()
    plt.close(fig)
    print(f"[graficos] Guardado: {ruta}")


def _explotar_generos(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por (cancion, genero); track_genre viene como 'a|b|c'."""
    return df.assign(genero=df["track_genre"].str.split("|")).explode("genero")


# ── EDA ───────────────────────────────────────────────────────────────────────

def grafico_generos_por_cancion(dist: pd.Series) -> None:
    """dist: canciones por numero de generos distintos (value_counts ordenado por indice)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(dist.index.astype(str), dist.values, color=SPOTIFY_GREEN, alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlabel("Numero de generos distintos por cancion")
    ax.set_ylabel("Canciones (escala log)")
    ax.set_title("Una misma cancion aparece en varios generos")
    fig.tight_layout()
    _save(fig, "00_generos_por_cancion.png")


def grafico_distribucion_popularidad(df: pd.DataFrame, umbral: float) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(df["popularity"], bins=50, color=SPOTIFY_GREEN, alpha=0.85)
    ax[0].axvline(umbral, color="#FFD700", ls="--", label=f"P80 (train) = {umbral:.0f}")
    ax[0].legend()
    ax[0].set_title("Distribucion de popularidad (train)")
    ax[1].hist(df.loc[df["popularity"] > 0, "popularity"], bins=50,
               color=SPOTIFY_GREEN, alpha=0.85)
    ax[1].axvline(umbral, color="#FFD700", ls="--")
    ax[1].set_title("Sin los casos con popularidad = 0")
    for a in ax:
        a.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    _save(fig, "01_distribucion_popularidad.png")


def grafico_distribuciones_numericas(df: pd.DataFrame, cols: list = None) -> None:
    cols = cols or NUM_COLS
    fig, axes = plt.subplots(2, 5, figsize=(18, 6))
    for ax, c in zip(axes.ravel(), cols):
        ax.hist(df[c].dropna(), bins=40, color=SPOTIFY_GREEN, alpha=0.85)
        ax.set_title(c, fontsize=10)
    fig.tight_layout()
    _save(fig, "02_distribuciones_numericas.png")


def tabla_hit_rate_genero(df: pd.DataFrame) -> pd.DataFrame:
    g = _explotar_generos(df)
    return (g.groupby("genero")["hit"].agg(["mean", "size"])
              .rename(columns={"mean": "hit_rate", "size": "n"})
              .sort_values("hit_rate"))


def grafico_hit_rate_genero(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    tasa = tabla_hit_rate_genero(df)
    global_rate = df["hit"].mean()
    extremos = pd.concat([tasa.head(n), tasa.tail(n)])
    fig, ax = plt.subplots(figsize=(8, 9))
    colores = [RED if v < global_rate else SPOTIFY_GREEN for v in extremos["hit_rate"]]
    ax.barh(extremos.index, extremos["hit_rate"], color=colores, alpha=0.85)
    ax.axvline(global_rate, color="black", ls="--", label=f"Global = {global_rate:.2f}")
    ax.set_title(f"Hit rate por genero (train): {n} mas bajos y {n} mas altos")
    ax.legend()
    fig.tight_layout()
    _save(fig, "03_hit_rate_por_genero.png")
    return tasa


def grafico_hit_rate_n_generos(df: pd.DataFrame) -> pd.DataFrame:
    t = (df.groupby("n_generos")["hit"].agg(["mean", "size"])
           .rename(columns={"mean": "hit_rate", "size": "n"}))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(t.index.astype(str), t["hit_rate"], color=SPOTIFY_GREEN, alpha=0.85)
    ax.axhline(df["hit"].mean(), color="black", ls="--")
    ax.set_xlabel("Numero de generos")
    ax.set_ylabel("Hit rate")
    ax.set_title("Hit rate segun numero de generos de la cancion")
    fig.tight_layout()
    _save(fig, "04_hit_rate_n_generos.png")
    return t.round(3)


def grafico_correlaciones(df: pd.DataFrame, cols: list = None) -> pd.DataFrame:
    cols = cols or CORR_COLS
    corr = df[cols + ["popularity", "hit"]].astype(float).corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                linewidths=0.5, linecolor="#0f0f1a", ax=ax,
                annot_kws={"size": 7}, cbar_kws={"shrink": .8})
    ax.set_title("Correlaciones (Pearson)")
    fig.tight_layout()
    _save(fig, "05_correlaciones.png")
    return corr


def grafico_features_por_clase(df: pd.DataFrame, cols: list = None) -> None:
    cols = cols or VARS_BOX
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, c in zip(axes.ravel(), cols):
        sns.boxplot(data=df, x="hit", y=c, ax=ax, showfliers=False,
                    palette=[RED, SPOTIFY_GREEN])
        ax.set_xticklabels(["No hit", "Hit"])
        ax.set_xlabel("")
        ax.set_title(c)
    fig.tight_layout()
    _save(fig, "06_features_por_clase.png")


def grafico_popularidad_cero(df: pd.DataFrame) -> pd.Series:
    g = _explotar_generos(df)
    cero = (g.assign(cero=(g["popularity"] == 0))
              .groupby("genero")["cero"].mean().sort_values(ascending=False))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(cero.values, bins=20, color=SPOTIFY_GREEN, alpha=0.85)
    ax.set_xlabel("Proporcion de canciones con popularidad 0 en el genero")
    ax.set_ylabel("Generos")
    ax.set_title("Popularidad 0: ruido uniforme o concentrado?")
    fig.tight_layout()
    _save(fig, "07_popularidad_cero.png")
    return cero


# ── Modelos ───────────────────────────────────────────────────────────────────

def grafico_roc(results: dict, y_test) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for (nombre, r), color in zip(results.items(), PALETTE):
        fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{nombre} (AUC={r['auc']:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Aleatorio (AUC=0.50)")
    ax.set_xlabel("Tasa de falsos positivos (FPR)")
    ax.set_ylabel("Tasa de verdaderos positivos (TPR)")
    ax.set_title("Curvas ROC — comparativa de modelos", fontsize=12, pad=12)
    ax.legend(fontsize=9)
    _save(fig, "08_curvas_roc.png")


def grafico_pr(results: dict, y_test) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    base = float(np.mean(y_test))
    for (nombre, r), color in zip(results.items(), PALETTE):
        prec, rec, _ = precision_recall_curve(y_test, r["y_proba"])
        ap = average_precision_score(y_test, r["y_proba"])
        ax.plot(rec, prec, color=color, linewidth=2, label=f"{nombre} (PR-AUC={ap:.4f})")
    ax.axhline(base, color="black", ls="--", linewidth=1, alpha=0.5,
               label=f"Base = {base:.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curvas Precision-Recall — comparativa de modelos", fontsize=12, pad=12)
    ax.legend(fontsize=9)
    _save(fig, "09_curvas_pr.png")


def grafico_matrices_confusion(results: dict, y_test) -> None:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (nombre, r) in zip(axes, results.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test, r["y_pred"], ax=ax,
            display_labels=["No-hit", "Hit"], colorbar=False, cmap="Greens"
        )
        ax.set_title(nombre, fontsize=10)
    fig.suptitle("Matrices de confusion — conjunto de prueba", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, "10_matrices_confusion.png")


def grafico_importancia_rf(pipe, top_n: int = 20) -> None:
    """Importancia de variables agrupada por variable original (Random Forest)."""
    rf = pipe.named_steps["clf"]
    importancias = rf.feature_importances_

    pre = pipe.named_steps["pre"]
    ohe_cols   = list(pre.named_transformers_["ohe"].get_feature_names_out(["key", "time_signature"]))
    scale_cols = list(pre.transformers_[0][2])
    pass_cols  = list(pre.transformers_[2][2])
    all_cols   = scale_cols + ohe_cols + pass_cols

    df_imp = pd.DataFrame({"feature": all_cols, "importancia": importancias})

    def grupo(f):
        if f.startswith("genre_"): return "Genero (multi-hot)"
        if f.startswith("key_"):   return "key (OHE)"
        if f.startswith("time_"):  return "time_signature (OHE)"
        return f
    df_imp["grupo"] = df_imp["feature"].apply(grupo)
    df_grp = df_imp.groupby("grupo")["importancia"].sum().sort_values(ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(df_grp.index, df_grp.values, color=SPOTIFY_GREEN, alpha=0.85)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8, color="#212529")
    ax.set_title("Importancia de variables — Random Forest (agrupada)", fontsize=12, pad=12)
    ax.set_xlabel("Importancia relativa")
    fig.tight_layout()
    _save(fig, "11_feature_importance_rf.png")


def grafico_ganancia_acumulada(results: dict, y_test) -> None:
    y = np.asarray(y_test)
    n = len(y)
    pct  = np.arange(1, n + 1) / n * 100
    base = np.cumsum(np.sort(y)[::-1]) / y.sum() * 100
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(pct, base, "k--", linewidth=1, alpha=0.4, label="Modelo perfecto")
    ax.plot([0, 100], [0, 100], "k:", linewidth=1, alpha=0.3, label="Aleatorio")
    for (nombre, r), color in zip(results.items(), PALETTE):
        idx  = np.argsort(r["y_proba"])[::-1]
        gain = np.cumsum(y[idx]) / y.sum() * 100
        ax.plot(pct, gain, color=color, linewidth=2,
                label=f"{nombre} (Lift@10%={r['lift_top10']:.2f}x)")
    ax.axvline(10, color="#FFD700", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("% de canciones evaluadas (ordenadas por probabilidad)")
    ax.set_ylabel("% de hits capturados")
    ax.set_title("Curva de ganancia acumulada — top-10% marcado", fontsize=12, pad=12)
    ax.legend(fontsize=9)
    _save(fig, "12_curva_ganancia_acumulada.png")


# ── Analisis de modelos (13 a 15; los usa 2_Modelacion.ipynb) ────────────────

def tabla_recall_por_genero(raw_test: pd.DataFrame, y_test, y_pred) -> pd.DataFrame:
    """
    Recall y precision por genero. Una cancion cuenta en cada uno de sus generos.
    raw_test debe estar en el mismo orden que y_test e y_pred.
    """
    d = _explotar_generos(
        raw_test[["track_genre"]].assign(y=np.asarray(y_test), pred=np.asarray(y_pred)))

    def resumen(s):
        hits = int(s["y"].sum())
        marcados = int(s["pred"].sum())
        return pd.Series({
            "n": len(s), "hits": hits,
            "recall": s["pred"][s["y"] == 1].mean() if hits > 0 else np.nan,
            "precision": s["y"][s["pred"] == 1].mean() if marcados > 0 else np.nan,
        })
    return d.groupby("genero").apply(resumen)


def grafico_recall_por_genero(elegibles: pd.DataFrame, meta: float = 0.75,
                              min_hits: int = 20) -> None:
    """elegibles: salida de tabla_recall_por_genero ya filtrada (hits >= min_hits, sin NaN)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(elegibles["recall"], bins=20, color=SPOTIFY_GREEN, alpha=0.85)
    ax.axvline(meta, color=RED, ls="--", label=f"KPI recall = {meta}")
    ax.legend()
    ax.set_xlabel("Recall del genero")
    ax.set_ylabel("Generos")
    ax.set_title(f"Recall por genero (generos con >= {min_hits} hits)")
    fig.tight_layout()
    _save(fig, "13_recall_por_genero.png")


def tabla_importancia_permutacion(modelo, X, y, n_muestra: int = 5000,
                                  n_rep: int = 3, seed: int = 42) -> pd.Series:
    """
    Caida del AUC al permutar cada variable en una muestra de X. Las columnas
    genre_* se permutan juntas, como un solo bloque. Devuelve la serie ordenada
    de menor a mayor caida.
    """
    from src.pipeline import FEATURES_SCALE, FEATURES_CAT, FEATURES_BIN, GENRE_PREFIX
    genre_cols = [c for c in X.columns if c.startswith(GENRE_PREFIX)]
    Xs = X.sample(n_muestra, random_state=seed)
    ys = y.loc[Xs.index]
    base = roc_auc_score(ys, modelo.predict_proba(Xs)[:, 1])

    bloques = {c: [c] for c in FEATURES_SCALE + FEATURES_CAT + FEATURES_BIN}
    bloques[f"género ({len(genre_cols)} columnas)"] = genre_cols

    rng = np.random.RandomState(seed)
    caida = {}
    for nombre, cols in bloques.items():
        c = []
        for _ in range(n_rep):
            Xp = Xs.copy()
            Xp[cols] = Xp[cols].values[rng.permutation(len(Xp))]
            c.append(base - roc_auc_score(ys, modelo.predict_proba(Xp)[:, 1]))
        caida[nombre] = np.mean(c)
    return pd.Series(caida).sort_values()


def grafico_importancia_permutacion(imp: pd.Series, nombre_modelo: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp.index, imp.values, color=SPOTIFY_GREEN, alpha=0.85)
    ax.set_xlabel("Caida del AUC al permutar la variable")
    ax.set_title(f"Importancia por permutacion - {nombre_modelo}")
    fig.tight_layout()
    _save(fig, "14_importancia_permutacion.png")


def silhouette_kmeans(X, ks=range(2, 9), n_muestra: int = 8000, seed: int = 42) -> dict:
    """Silhouette de K-Means para cada k, calculado sobre una muestra de X (array escalado)."""
    muestra = np.random.RandomState(seed).choice(len(X), n_muestra, replace=False)
    sil = {}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        sil[k] = silhouette_score(X[muestra], km.labels_[muestra])
    return sil


def grafico_kmeans_silhouette(sil: dict) -> int:
    """Grafica silhouette vs k y devuelve el k con mayor silhouette."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(list(sil), list(sil.values()), marker="o", color=SPOTIFY_GREEN)
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette")
    ax.set_title("Seleccion de k (K-Means)")
    fig.tight_layout()
    _save(fig, "15_kmeans_silhouette.png")
    return max(sil, key=sil.get)


# ── Funcion principal ─────────────────────────────────────────────────────────

def grafico_politica_umbral(recall_tramos: pd.DataFrame, marcadas_n: pd.DataFrame,
                            meta: float = 0.75) -> None:
    """
    Evidencia de la politica de umbral (seccion 10 de 2_Modelacion), umbral 0.5 vs oficial.
    recall_tramos: indice = tramo de hit rate del genero, columnas = politicas (recall en test).
    marcadas_n:    indice = 1, 2, 3+ generos, columnas = politicas (marcadas / hit rate real).
    """
    colores = {"umbral 0.5": "#9E9E9E", "oficial": SPOTIFY_GREEN}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    paneles = [(axes[0], recall_tramos, "Recall por tramo de hit rate del género", "Recall (test)", "{:.2f}"),
               (axes[1], marcadas_n, "Marcadas / hit rate real, por n° de géneros", "Veces su tasa real de hits", "{:.2f}")]
    for ax, df_p, titulo, ylab, fmt in paneles:
        x = np.arange(len(df_p.index)); ancho = 0.38
        for k, col in enumerate(df_p.columns):
            barras = ax.bar(x + (k - 0.5) * ancho, df_p[col].values, width=ancho - 0.03,
                            color=colores.get(col, PALETTE[k]), label=col)
            ax.bar_label(barras, labels=[fmt.format(v) for v in df_p[col].values],
                         padding=2, fontsize=8, color="#212529")
        ax.set_xticks(x); ax.set_xticklabels([str(i) for i in df_p.index])
        ax.set_title(titulo, fontsize=11, pad=10); ax.set_ylabel(ylab)
        ax.grid(axis="y"); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].axhline(meta, color=RED, ls="--", lw=1, label=f"KPI recall = {meta}")
    axes[0].set_ylim(0, 1.12); axes[0].set_xlabel("Hit rate del género")
    axes[1].set_ylim(0, max(2.6, float(marcadas_n.values.max()) * 1.15)); axes[1].set_xlabel("Géneros de la canción")
    axes[0].legend(fontsize=8, loc="upper left"); axes[1].legend(fontsize=8, loc="upper left")
    fig.suptitle("Política de umbral: umbral 0.5 frente a la política oficial (test)", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, "20_politica_umbral.png")


def _cargar_eda():
    """
    Mismo df que usa 1_EDA.ipynb: data/raw/train/raw_train.csv con las
    transformaciones aplicadas (log1p + hit). El umbral P80 se calcula SOLO con train.
    """
    from src.pipeline import RAW_TRAIN_DIR, aplicar_transformaciones
    ruta = os.path.join(RAW_TRAIN_DIR, "raw_train.csv")
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No existe {ruta}. Ejecuta 'python -m src.pipeline' primero.")
    df, umbral = aplicar_transformaciones(pd.read_csv(ruta))
    return df, umbral


def run(results: dict = None, y_test=None, rf_pipe=None) -> None:
    """Genera todos los graficos. Los de modelos solo si se pasan results y y_test."""
    print("=" * 60)
    print("GRAFICOS — Generando visualizaciones")
    print("=" * 60)

    df, umbral = _cargar_eda()
    grafico_distribucion_popularidad(df, umbral)
    grafico_distribuciones_numericas(df)
    grafico_hit_rate_genero(df)
    grafico_hit_rate_n_generos(df)
    grafico_correlaciones(df)
    grafico_features_por_clase(df)
    grafico_popularidad_cero(df)

    if results and y_test is not None:
        # El modelo oficial tiene las mismas probabilidades que su modelo base: las curvas
        # (ROC, PR, ganancia) se dibujan una vez por modelo; la matriz de confusion lo incluye.
        curvas = {k: v for k, v in results.items() if not k.endswith("(oficial)")}
        grafico_roc(curvas, y_test)
        grafico_pr(curvas, y_test)
        grafico_matrices_confusion(results, y_test)
        grafico_ganancia_acumulada(curvas, y_test)
        if rf_pipe is not None:
            grafico_importancia_rf(rf_pipe)

    print("\n[graficos] Visualizaciones completadas.")


if __name__ == "__main__":
    run()
