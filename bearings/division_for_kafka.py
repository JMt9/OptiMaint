import os
import numpy as np
import pandas as pd


# CONFIGURACION:
CSV_5P = "data/features5.csv"
CSV_10P = "data/features10.csv"

OUT_DIR = "data"
TRAIN_OUT = "features_train.csv" # 85% -> desde features10
KAFKA_OUT = "features_kafka.csv" # 15% -> desde features5

SEED = 42
TEST_SIZE = 0.15 # para kafka

FAULT_COL = "fault"
SUBFAULT_COL = "fault2"
ID_COL = "id"


def build_strata_key(df: pd.DataFrame) -> pd.Series:
    """
    Función para hacer las 10 clases para hacer la división train/kafka de manera representativa:
    Clave de estratificación en 10 clases:
    - Si fault es overhang/underhang => fault__fault2
    - Si no => fault
    """
    if FAULT_COL not in df.columns:
        raise ValueError(f"Falta '{FAULT_COL}' para estratificar.")

    fault = df[FAULT_COL].astype(str)

    if SUBFAULT_COL in df.columns:
        sub = df[SUBFAULT_COL].astype("string")
    else:
        sub = pd.Series([pd.NA] * len(df), index=df.index, dtype="string")

    is_hang = fault.isin(["overhang", "underhang"])

    key = fault.copy()
    key.loc[is_hang] = fault.loc[is_hang] + "__" + sub.loc[is_hang].fillna("MISSING_SUBLABEL")
    return key


def stratified_split_by_key(df: pd.DataFrame, key: pd.Series, test_size: float, seed: int):
    """
    Split estratificado manual por 'key'.
    - Para cada estrato: baraja índices y mete ~test_size en test
    - Asegura al menos 1 en test y 1 en train si el estrato tiene >=2
    - Si estrato tiene <2, va entero a train (y avisa)
    """
    rng = np.random.default_rng(seed)

    counts = key.value_counts()
    too_small = counts[counts < 2]
    if len(too_small) > 0:
        print("\n[WARN] Estratos con <2 muestras -> se quedan SOLO en TRAIN:")
        for k, v in too_small.items():
            print(f"  - {k}: {v}")

    train_idx = []
    test_idx = []

    grouped = df.groupby(key, sort=False).groups # dict: key -> index list

    for k, idxs in grouped.items():
        idxs = np.array(list(idxs))
        rng.shuffle(idxs)

        n = len(idxs)
        if n < 2:
            train_idx.extend(idxs.tolist())
            continue

        n_test = int(np.floor(test_size * n))
        n_test = max(1, n_test)
        n_test = min(n_test, n - 1)

        test_part = idxs[:n_test]
        train_part = idxs[n_test:]

        test_idx.extend(test_part.tolist())
        train_idx.extend(train_part.tolist())

    train_df = df.loc[train_idx].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = df.loc[test_idx].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_df, test_df


# MAIN
def main():
    if not os.path.exists(CSV_5P):
        raise FileNotFoundError(f"No existe el CSV: {CSV_5P}")
    if not os.path.exists(CSV_10P):
        raise FileNotFoundError(f"No existe el CSV: {CSV_10P}")

    df5 = pd.read_csv(CSV_5P)
    df10 = pd.read_csv(CSV_10P)

    # checks
    if FAULT_COL not in df5.columns or FAULT_COL not in df10.columns:
        raise ValueError(f"Falta '{FAULT_COL}' en alguno de los CSVs.")
    if ID_COL in df5.columns or ID_COL in df10.columns:
        raise ValueError(f"Alguno de los CSVs ya tiene columna '{ID_COL}'. Quítala del input.")

    # si el orden es el mismo, tiene que haber misma longitud
    if len(df5) != len(df10):
        raise ValueError(
            "Las longitudes no coinciden por lo que no se puede emparejar por índice.\n"
            f"  - len(features5)={len(df5)}\n"
            f"  - len(features10)={len(df10)}\n"
        )

    n = len(df10)

    # Añadimos id por posición (misma muestra = mismo índice)
    ids = np.arange(n, dtype=np.int64)
    df5 = df5.copy()
    df10 = df10.copy()
    df5.insert(0, ID_COL, ids)
    df10.insert(0, ID_COL, ids)

    # Split estratificado en df10 (equivalente a hacerlo en df5)
    key = build_strata_key(df10)
    train10, test10 = stratified_split_by_key(df10, key, test_size=TEST_SIZE, seed=SEED)

    train_ids = set(train10[ID_COL].tolist())
    test_ids = set(test10[ID_COL].tolist())
    all_ids = set(ids.tolist())

    # checks
    if train_ids & test_ids:
        raise RuntimeError("[SPLIT] Solape de IDs entre train y kafka (no debería).")
    if (train_ids | test_ids) != all_ids:
        missing = all_ids - (train_ids | test_ids)
        raise RuntimeError(f"[SPLIT] No estás usando el 100% de IDs. Missing: {len(missing)}")

    # Construimos outputs cruzados por id
    features_train = df10[df10[ID_COL].isin(train_ids)].copy()  # 85% desde 10p
    features_kafka = df5[df5[ID_COL].isin(test_ids)].copy()    # 15% desde 5p

    # barajar reproducible
    features_train = features_train.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    features_kafka = features_kafka.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    print("\n=== SPLIT ===")
    print("Total:", n)
    print("Train rows:", len(features_train), f"({len(features_train)/n*100:.2f}%)  -> features10")
    print("Kafka rows:", len(features_kafka), f"({len(features_kafka)/n*100:.2f}%)  -> features5")

    # Guardamos
    os.makedirs(OUT_DIR, exist_ok=True)
    train_path = os.path.join(OUT_DIR, TRAIN_OUT)
    kafka_path = os.path.join(OUT_DIR, KAFKA_OUT)

    features_train.to_csv(train_path, index=False)
    features_kafka.to_csv(kafka_path, index=False)

    print("\n=== SAVED ===")
    print("Train:", train_path)
    print("Kafka:", kafka_path)
    print("\nDONE")


if __name__ == "__main__":
    main()
