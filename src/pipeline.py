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
# Variables baratas del texto (seccion 13 del notebook): sin identidad del artista
FEATURES_TEXT_NUM = ["n_artistas", "t_palabras"]
FEATURES_TEXT_BIN = ["t_remix", "t_live", "t_feat"]
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
    # 1 fila con nulos en artists, album_name y track_name: se descarta
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

def variables_texto(df):
    """
    Variables del texto que no identifican al artista: numero de artistas (colaboraciones)
    y marcas en el titulo (remix, live, feat., cantidad de palabras). No usan el target.
    """
    t = df["track_name"].fillna("").str.lower()
    return pd.DataFrame({
        "n_artistas": df["artists"].fillna("").str.count(";") + 1,
        "t_remix":    t.str.contains(r"remix|\bmix\b", regex=True).astype(int),
        "t_live":     t.str.contains(r"\blive\b|en vivo", regex=True).astype(int),
        "t_feat":     t.str.contains(r"feat\.?|\bft\.|\bwith\b", regex=True).astype(int),
        "t_palabras": t.str.split().str.len().fillna(0).astype(int),
    }, index=df.index)

def aplicar_transformaciones(df, threshold=None):
    """ Aplica log1p, agrega variables de texto y define target (hit) """
    df = df.copy()
    log_vars = ["speechiness", "instrumentalness", "liveness", "duration_ms"]
    for v in log_vars:
        df[f"{v}_log"] = np.log1p(df[v])
    df = pd.concat([df, variables_texto(df)], axis=1)
    
    if threshold is None:
        threshold = df["popularity"].quantile(HIT_PERCENTILE)
        print(f"[pipeline] Umbral de popularidad (P80) calculado en TRAIN: {threshold}")
        
    df[TARGET] = (df["popularity"] >= threshold).astype(int)
    return df, threshold

def guardar_processed(train, test, genre_cols):
    print("[pipeline] Guardando datos procesados...")
    os.makedirs(PROC_TRAIN_DIR, exist_ok=True)
    os.makedirs(PROC_TEST_DIR, exist_ok=True)

    ALL_FEATURES = FEATURES_SCALE + FEATURES_CAT + FEATURES_BIN + FEATURES_TEXT_NUM + FEATURES_TEXT_BIN + genre_cols

    # Train
    train[ALL_FEATURES].to_csv(os.path.join(PROC_TRAIN_DIR, "X_train.csv"), index=False)
    train[[TARGET]].to_csv(os.path.join(PROC_TRAIN_DIR, "y_train.csv"), index=False)
    train[["song_key"]].to_csv(os.path.join(PROC_TRAIN_DIR, "g_train.csv"), index=False)
    train[["track_id"]].to_csv(os.path.join(PROC_TRAIN_DIR, "id_train.csv"), index=False)

    # Test
    test[ALL_FEATURES].to_csv(os.path.join(PROC_TEST_DIR, "X_test.csv"), index=False)
    test[[TARGET]].to_csv(os.path.join(PROC_TEST_DIR, "y_test.csv"), index=False)
    test[["track_id"]].to_csv(os.path.join(PROC_TEST_DIR, "id_test.csv"), index=False)

    pd.Series(ALL_FEATURES).to_csv(os.path.join("data", "processed", "feature_names.csv"), index=False, header=["feature"])

    print(f"[pipeline] Processed Train: {len(train)}, Processed Test: {len(test)}")

# ----------------------------------------------------------------------------------------------
# ETAPAS DEL PIPELINE
#   Etapa 1 (inicio del EDA): generar_split_raw()  -> data/raw/train|test
#   Etapa 2 (cierre del EDA): generar_processed()  -> data/processed/train|test
#   Modelacion: NO divide ni transforma; solo consume data/processed (verificar_processed()).
# El split processed son LAS MISMAS FILAS (mismo orden) que el split raw, ya transformadas.
# ----------------------------------------------------------------------------------------------

def _rutas_raw():
    return (os.path.join(RAW_TRAIN_DIR, "raw_train.csv"),
            os.path.join(RAW_TEST_DIR, "raw_test.csv"))

def generar_split_raw(forzar=False):
    """
    Etapa 1: limpieza + deduplicacion agregada + multi-hot + split agrupado por song_key.
    Guarda data/raw/train/raw_train.csv y data/raw/test/raw_test.csv.
    Si ya existen y forzar=False se reutilizan (el split es determinista, random_state=42);
    usa forzar=True si cambiaste la limpieza o el split.
    """
    ruta_tr, ruta_te = _rutas_raw()
    if not forzar and os.path.exists(ruta_tr) and os.path.exists(ruta_te):
        print("[pipeline] Split raw ya existe: se reutiliza (forzar=True para regenerarlo).")
        return pd.read_csv(ruta_tr), pd.read_csv(ruta_te)

    df_raw = cargar_datos_raw()
    df_clean, genre_cols, mlb = pre_procesar_y_deduplicar(df_raw)
    return split_raw_data(df_clean)

def generar_processed():
    """
    Etapa 2: lee SOLO raw_train.csv y raw_test.csv (no vuelve a dividir), aplica log1p, calcula
    el umbral P80 con TRAIN, define el target y guarda data/processed/train|test.
    El StandardScaler NO se aplica aqui: va dentro del Pipeline de train.py (se ajusta en cada fold).
    """
    ruta_tr, ruta_te = _rutas_raw()
    if not (os.path.exists(ruta_tr) and os.path.exists(ruta_te)):
        raise FileNotFoundError("Faltan data/raw/train|test. Ejecuta antes generar_split_raw() "
                                "(inicio del notebook 1_EDA o python main.py).")
    train_raw = pd.read_csv(ruta_tr)
    test_raw = pd.read_csv(ruta_te)

    genre_cols = [c for c in train_raw.columns if c.startswith(GENRE_PREFIX)]
    assert genre_cols == [c for c in test_raw.columns if c.startswith(GENRE_PREFIX)], \
        "Las columnas genre_* de raw train y raw test no coinciden"
    sin_train = [c for c in genre_cols if train_raw[c].sum() == 0]
    if sin_train:
        print(f"[pipeline] AVISO: {len(sin_train)} generos sin canciones en train (columnas todo 0): {sin_train[:5]}")

    print("[pipeline] Aplicando transformaciones a Train...")
    train_proc, threshold = aplicar_transformaciones(train_raw)
    print("[pipeline] Aplicando transformaciones a Test...")
    test_proc, _ = aplicar_transformaciones(test_raw, threshold=threshold)

    guardar_processed(train_proc, test_proc, genre_cols)

    # Trazabilidad: processed son las mismas filas, en el mismo orden, que raw
    ids_tr = pd.read_csv(os.path.join(PROC_TRAIN_DIR, "id_train.csv"))["track_id"]
    ids_te = pd.read_csv(os.path.join(PROC_TEST_DIR, "id_test.csv"))["track_id"]
    assert len(ids_tr) == len(train_raw) and (ids_tr.values == train_raw["track_id"].values).all(), "processed train no coincide con raw train"
    assert len(ids_te) == len(test_raw) and (ids_te.values == test_raw["track_id"].values).all(), "processed test no coincide con raw test"
    print("[pipeline] Trazabilidad OK: processed = mismas filas que raw (mismo orden).")
    return {"train_proc": train_proc, "test_proc": test_proc, "genre_cols": genre_cols, "threshold": threshold}

def verificar_processed():
    """Para Modelacion: comprueba que existe data/processed; si no, avisa que hay que correr el EDA/main."""
    req = [os.path.join(PROC_TRAIN_DIR, f) for f in ("X_train.csv", "y_train.csv", "g_train.csv")] + \
          [os.path.join(PROC_TEST_DIR, f) for f in ("X_test.csv", "y_test.csv")]
    faltan = [r for r in req if not os.path.exists(r)]
    if faltan:
        raise FileNotFoundError("Falta data/processed (" + ", ".join(os.path.basename(f) for f in faltan) +
                                "). Ejecuta el notebook 1_EDA completo o python main.py.")
    print("[pipeline] data/processed disponible.")

def run(forzar_raw=False):
    """Etapa 1 + etapa 2 (lo usa main.py)."""
    print("=" * 60)
    print("PIPELINE - Preprocesamiento Spotify (Raw & Processed)")
    print("=" * 60)
    generar_split_raw(forzar=forzar_raw)
    res = generar_processed()
    print("[pipeline] Finalizado exitosamente.")
    return res

if __name__ == "__main__":
    run()
