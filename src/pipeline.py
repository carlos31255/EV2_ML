"""
src/pipeline.py
---------------
Preprocesamiento completo del dataset de Spotify.

Flujo (segun estructura oficial del docente):
  Raw Data
  -> Seleccion de columnas (eliminar baja varianza, muchos nulos)
  -> Tipos de variable
  -> NA (eliminacion)
  -> Eliminar duplicados (exactos)
  -> Transformaciones (log1p en variables asimetricas)
  -> Transformacion multiclase del genero (multi-hot, 114 columnas)
  -> Definicion de variable objetivo (hit/no-hit, P80 solo sobre train)
  -> Split train/test por cancion (GroupShuffleSplit)
  -> Guardar data/processed/train/ y data/processed/test/

Uso:
  python -m src.pipeline          # desde linea de comandos
  from src.pipeline import run    # desde otro modulo
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer

# ── Constantes ──────────────────────────────────────────────────────────────

RAW_CSV         = os.path.join("data", "processed", "spotify_clean.csv")
TRAIN_DIR       = os.path.join("data", "processed", "train")
TEST_DIR        = os.path.join("data", "processed", "test")

SEED            = 42
TEST_SIZE       = 0.20
HIT_PERCENTILE  = 0.80

# Variables por tipo (definicion explicita segun distribucion, EDA EV1)
FEATURES_SCALE = [          # Continuas: StandardScaler (tras log1p, distribuciones menos asimetricas)
    "danceability", "energy", "loudness", "valence", "tempo", "acousticness",
    "speechiness_log", "instrumentalness_log", "liveness_log", "duration_ms_log"
]
FEATURES_CAT   = ["key", "time_signature"]   # Nominales: OneHotEncoder
FEATURES_BIN   = ["explicit", "mode"]         # Binarias: sin transformar
# FEATURES_GENRE se construye dinamicamente tras el multi-hot

# Columnas de identificacion (no entran al modelo)
ID_COLS = ["track_id", "track_name", "artists", "album_name", "track_genre",
           "track_genre_encoded", "song_key"]
TARGET  = "hit"


# ── Funciones ────────────────────────────────────────────────────────────────

def cargar_datos(ruta: str = RAW_CSV) -> pd.DataFrame:
    """Carga el dataset limpio generado por EV1."""
    print(f"[pipeline] Cargando datos desde {ruta} ...")
    df = pd.read_csv(ruta)
    print(f"[pipeline] Shape cargado: {df.shape}")
    return df


def seleccionar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina columnas de identificacion que no aportan informacion predictiva.
    Varianza 0 y columnas con >50% de nulos (ninguna en este dataset).
    """
    # Columnas con varianza 0
    zero_var = [c for c in df.select_dtypes(include="number").columns
                if df[c].var() == 0]
    # Columnas con >50% nulos
    many_nulls = [c for c in df.columns if df[c].isnull().mean() > 0.50]

    drop = list(set(zero_var + many_nulls))
    if drop:
        print(f"[pipeline] Columnas eliminadas (var=0 o >50% nulos): {drop}")
        df = df.drop(columns=drop)
    else:
        print("[pipeline] Sin columnas a eliminar por varianza o nulos.")
    return df


def manejar_nulos(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas con nulos en variables clave (1 fila en EV1)."""
    antes = len(df)
    df = df.dropna(subset=["artists", "album_name", "track_name"])
    df = df[df["duration_ms"] > 0]
    print(f"[pipeline] Filas eliminadas por nulos/duracion invalida: {antes - len(df)}")
    return df


def eliminar_duplicados(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina duplicados exactos (450 filas en EV1)."""
    antes = len(df)
    df = df.drop_duplicates()
    print(f"[pipeline] Duplicados exactos eliminados: {antes - len(df)}")
    return df


def transformar_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica log1p a variables con alta asimetria positiva identificadas en EDA.
    Si las columnas *_log ya existen (vienen de EV1), se reutilizan.
    """
    log_vars = ["speechiness", "instrumentalness", "liveness", "duration_ms"]
    for v in log_vars:
        col_log = f"{v}_log"
        if col_log not in df.columns:
            df[col_log] = np.log1p(df[v])
            print(f"[pipeline] Creada columna {col_log}")
        else:
            print(f"[pipeline] Reutilizando columna {col_log} existente")
    return df


def genero_multihot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transformacion multiclase del genero:
    - Una fila por track_id (promedio de popularity si difiere entre filas).
    - 114 columnas genre_* (multi-hot).
    - Crea song_key = artista + nombre de pista en minusculas.
    """
    print("[pipeline] Construyendo song_key ...")
    df["song_key"] = (
        df["artists"].str.lower().str.strip() + "||" +
        df["track_name"].str.lower().str.strip()
    )

    print("[pipeline] Agrupando generos por track_id ...")
    generos = (
        df.groupby("track_id")["track_genre"]
        .apply(lambda s: sorted(set(s)))
        .to_dict()
    )

    print("[pipeline] Deduplicando a una fila por track_id ...")
    # Popularity puede diferir entre filas de un mismo track_id (0.8% de canciones)
    df_pop = df.groupby("track_id")["popularity"].mean().round().astype(int)
    df_song = df.drop_duplicates(subset=["track_id"]).copy()
    df_song["popularity"] = df_song["track_id"].map(df_pop)

    print("[pipeline] Aplicando MultiLabelBinarizer para multi-hot ...")
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(df_song["track_id"].map(generos))
    genre_cols   = [f"genre_{g}" for g in mlb.classes_]
    df_genres    = pd.DataFrame(genre_matrix, columns=genre_cols, index=df_song.index)
    df_song      = pd.concat([df_song, df_genres], axis=1)

    n_canciones = len(df_song)
    print(f"[pipeline] Canciones unicas tras deduplicacion: {n_canciones:,}")
    print(f"[pipeline] Columnas genre_*: {len(genre_cols)}")
    return df_song, mlb


def split_train_test(df_song: pd.DataFrame) -> tuple:
    """
    Split 80/20 por cancion (GroupShuffleSplit sobre song_key).
    El umbral P80 se calcula SOLO con train para evitar fuga de datos.
    """
    print("[pipeline] Realizando split train/test por cancion ...")
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_idx, test_idx = next(gss.split(df_song, groups=df_song["song_key"]))
    train = df_song.iloc[train_idx].copy()
    test  = df_song.iloc[test_idx].copy()

    # Verificacion: ninguna cancion en ambos conjuntos
    songs_train = set(train["song_key"])
    songs_test  = set(test["song_key"])
    assert len(songs_train & songs_test) == 0, "ERROR: hay canciones en train y test!"
    print(f"[pipeline] Train: {len(train):,} canciones | Test: {len(test):,} canciones")

    # P80 calculado SOLO con train
    threshold = train["popularity"].quantile(HIT_PERCENTILE)
    train[TARGET] = (train["popularity"] >= threshold).astype(int)
    test[TARGET]  = (test["popularity"]  >= threshold).astype(int)
    print(f"[pipeline] Umbral P{int(HIT_PERCENTILE*100)} (calculado en train): {threshold:.0f} pts")
    print(f"[pipeline] Hit rate train: {train[TARGET].mean()*100:.1f}% | test: {test[TARGET].mean()*100:.1f}%")
    return train, test


def guardar_splits(train: pd.DataFrame, test: pd.DataFrame,
                   features_genre: list) -> None:
    """Guarda los conjuntos de datos en data/processed/train/ y test/."""
    ALL_FEATURES = FEATURES_SCALE + FEATURES_CAT + FEATURES_BIN + features_genre

    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR,  exist_ok=True)

    # Train
    train[ALL_FEATURES].to_csv(os.path.join(TRAIN_DIR, "X_train.csv"), index=False)
    train[[TARGET]].to_csv(     os.path.join(TRAIN_DIR, "y_train.csv"), index=False)
    train[["song_key"]].to_csv( os.path.join(TRAIN_DIR, "g_train.csv"), index=False)

    # Test
    test[ALL_FEATURES].to_csv(os.path.join(TEST_DIR, "X_test.csv"), index=False)
    test[[TARGET]].to_csv(     os.path.join(TEST_DIR, "y_test.csv"), index=False)

    # Metadata
    pd.Series(ALL_FEATURES).to_csv(
        os.path.join("data", "processed", "feature_names.csv"), index=False, header=["feature"]
    )

    print(f"[pipeline] Guardado: {TRAIN_DIR}/X_train.csv ({train[ALL_FEATURES].shape})")
    print(f"[pipeline] Guardado: {TEST_DIR}/X_test.csv ({test[ALL_FEATURES].shape})")
    print(f"[pipeline] Features totales: {len(ALL_FEATURES)}")


# ── Funcion principal ────────────────────────────────────────────────────────

def run() -> dict:
    """Ejecuta el pipeline completo y devuelve el contexto para otros modulos."""
    print("=" * 60)
    print("PIPELINE — Preprocesamiento Spotify")
    print("=" * 60)

    df = cargar_datos()
    df = seleccionar_columnas(df)
    df = manejar_nulos(df)
    df = eliminar_duplicados(df)
    df = transformar_variables(df)
    df_song, mlb = genero_multihot(df)
    train, test  = split_train_test(df_song)

    features_genre = [c for c in df_song.columns if c.startswith("genre_")]
    guardar_splits(train, test, features_genre)

    print("\n[pipeline] Pipeline completado exitosamente.")
    return {
        "train": train, "test": test,
        "features_genre": features_genre, "mlb": mlb
    }


if __name__ == "__main__":
    run()
