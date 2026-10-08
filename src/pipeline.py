import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer

# Directorios
RAW_TRAIN_DIR  = os.path.join("data", "raw", "train")
RAW_TEST_DIR   = os.path.join("data", "raw", "test")
PROC_TRAIN_DIR = os.path.join("data", "processed", "train")
PROC_TEST_DIR  = os.path.join("data", "processed", "test")

SEED = 42
TEST_SIZE = 0.20
HIT_PERCENTILE = 0.80

FEATURES_SCALE = ["danceability", "energy", "loudness", "valence", "tempo", "acousticness", "speechiness_log", "instrumentalness_log", "liveness_log", "duration_ms_log"]
FEATURES_CAT = ["key", "time_signature"]
FEATURES_BIN = ["explicit", "mode"]
TARGET = "hit"
GENRE_PREFIX = "genre_"

def cargar_datos_raw():
    print("[pipeline] Cargando raw data...")
    return pd.read_csv(os.path.join("data", "raw", "Spotify_Tracks_Dataset.csv"))

def pre_procesar_basico(df):
    """
    Limpieza + deduplicacion agregada: una fila por track_id, sin perder generos.
    Sin multi-hot (eso lo hace pre_procesar_y_deduplicar). Lo usa el EDA.

    Columnas resultantes relevantes:
      - popularity : maximo entre las copias del track_id
      - track_genre: todos los generos de la cancion unidos con '|'
      - n_generos  : cantidad de generos de la cancion
      - song_key   : artists + track_name normalizados (clave del split)
      - grupo      : alias de song_key (lo usa el notebook 1_EDA)
    """
    print("[pipeline] Eliminando nulos criticos...")
    # 3 nulos de texto, simplemente los descartamos
    df = df.dropna(subset=["artists", "album_name", "track_name", "track_id"]).copy()

    # Popularidad 0 no se borra.

    print("[pipeline] Construyendo song_key para el split (artists + track_name)...")
    df["song_key"] = df["artists"].str.lower().str.strip() + "||" + df["track_name"].str.lower().str.strip()

    print("[pipeline] Agrupando generos y calculando max popularity por track_id...")
    generos = df.groupby("track_id")["track_genre"].apply(lambda s: sorted(set(s)))
    df_pop = df.groupby("track_id")["popularity"].max().astype(int)

    df_unique = df.drop_duplicates(subset=["track_id"], keep="first").copy()
    df_unique["popularity"] = df_unique["track_id"].map(df_pop)
    lista = df_unique["track_id"].map(generos)
    df_unique["track_genre"] = lista.apply("|".join)
    df_unique["n_generos"] = lista.apply(len)
    df_unique["grupo"] = df_unique["song_key"]
    return df_unique

def pre_procesar_y_deduplicar(df):
    """ Deduplica agrupando por track_id para no perder multiples generos """
    df_unique = pre_procesar_basico(df)

    print("[pipeline] Aplicando MultiLabelBinarizer para generos...")
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(df_unique["track_genre"].str.split("|"))
    genre_cols = [f"{GENRE_PREFIX}{g}" for g in mlb.classes_]
    df_genres = pd.DataFrame(genre_matrix, columns=genre_cols, index=df_unique.index)

    df_unique = pd.concat([df_unique, df_genres], axis=1)
    print(f"[pipeline] Canciones unicas: {len(df_unique)}. Columnas genre_*: {len(genre_cols)}")
    return df_unique, genre_cols, mlb

def split_raw_data(df):
    print("[pipeline] Haciendo split agrupado (GroupShuffleSplit) sobre song_key...")
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_idx, test_idx = next(gss.split(df, groups=df["song_key"]))
    train_raw = df.iloc[train_idx].copy()
    test_raw = df.iloc[test_idx].copy()
    
    os.makedirs(RAW_TRAIN_DIR, exist_ok=True)
    os.makedirs(RAW_TEST_DIR, exist_ok=True)
    train_raw.to_csv(os.path.join(RAW_TRAIN_DIR, "raw_train.csv"), index=False)
    test_raw.to_csv(os.path.join(RAW_TEST_DIR, "raw_test.csv"), index=False)
    print(f"[pipeline] Raw Train: {len(train_raw)}, Raw Test: {len(test_raw)}")
    return train_raw, test_raw

def aplicar_transformaciones(df, threshold=None):
    """ Aplica log1p y define target (hit) """
    df = df.copy()
    log_vars = ["speechiness", "instrumentalness", "liveness", "duration_ms"]
    for v in log_vars:
        df[f"{v}_log"] = np.log1p(df[v])
    
    if threshold is None:
        threshold = df["popularity"].quantile(HIT_PERCENTILE)
        print(f"[pipeline] Umbral de popularidad (P80) calculado en TRAIN: {threshold}")
        
    df[TARGET] = (df["popularity"] >= threshold).astype(int)
    return df, threshold

def guardar_processed(train, test, genre_cols):
    print("[pipeline] Guardando datos procesados...")
    os.makedirs(PROC_TRAIN_DIR, exist_ok=True)
    os.makedirs(PROC_TEST_DIR, exist_ok=True)
    
    ALL_FEATURES = FEATURES_SCALE + FEATURES_CAT + FEATURES_BIN + genre_cols
    
    # Train
    train[ALL_FEATURES].to_csv(os.path.join(PROC_TRAIN_DIR, "X_train.csv"), index=False)
    train[[TARGET]].to_csv(os.path.join(PROC_TRAIN_DIR, "y_train.csv"), index=False)
    train[["song_key"]].to_csv(os.path.join(PROC_TRAIN_DIR, "g_train.csv"), index=False)
    
    # Test
    test[ALL_FEATURES].to_csv(os.path.join(PROC_TEST_DIR, "X_test.csv"), index=False)
    test[[TARGET]].to_csv(os.path.join(PROC_TEST_DIR, "y_test.csv"), index=False)
    
    pd.Series(ALL_FEATURES).to_csv(os.path.join("data", "processed", "feature_names.csv"), index=False, header=["feature"])
    
    print(f"[pipeline] Processed Train: {len(train)}, Processed Test: {len(test)}")

def run():
    print("=" * 60)
    print("PIPELINE - Preprocesamiento Spotify (Raw & Processed)")
    print("=" * 60)
    
    df_raw = cargar_datos_raw()
    df_clean, genre_cols, mlb = pre_procesar_y_deduplicar(df_raw)
    
    train_raw, test_raw = split_raw_data(df_clean)
    
    print("[pipeline] Aplicando transformaciones a Train...")
    train_proc, threshold = aplicar_transformaciones(train_raw)
    
    print("[pipeline] Aplicando transformaciones a Test...")
    test_proc, _ = aplicar_transformaciones(test_raw, threshold=threshold)
    
    guardar_processed(train_proc, test_proc, genre_cols)
    print("[pipeline] Finalizado exitosamente.")
    return {"train_proc": train_proc, "test_proc": test_proc, "genre_cols": genre_cols}

if __name__ == "__main__":
    run()