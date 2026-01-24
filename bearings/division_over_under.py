import os
import pandas as pd

# CONFIGURACIÓN
CSV_IN = "data/features_train.csv"
OUT_DIR = "data"

LABEL_COL = "fault"
OVER_LABEL = "overhang"
UNDER_LABEL = "underhang"

OUT_OVER = "features_overhang.csv"
OUT_UNDER = "features_underhang.csv"


def main():
    if not os.path.exists(CSV_IN):
        raise FileNotFoundError(f"No existe el CSV: {CSV_IN}")

    df = pd.read_csv(CSV_IN)

    if LABEL_COL not in df.columns:
        raise ValueError(f"Falta la columna '{LABEL_COL}' en {CSV_IN}")

    df[LABEL_COL] = df[LABEL_COL].astype(str)

    df_over = df[df[LABEL_COL] == OVER_LABEL].copy()
    df_under = df[df[LABEL_COL] == UNDER_LABEL].copy()

    if len(df_over) == 0:
        print(f"[WARN] No hay filas con {LABEL_COL} == '{OVER_LABEL}'")
    if len(df_under) == 0:
        print(f"[WARN] No hay filas con {LABEL_COL} == '{UNDER_LABEL}'")

    os.makedirs(OUT_DIR, exist_ok=True)

    out_over_path = os.path.join(OUT_DIR, OUT_OVER)
    out_under_path = os.path.join(OUT_DIR, OUT_UNDER)

    df_over.to_csv(out_over_path, index=False)
    df_under.to_csv(out_under_path, index=False)

    print("\n=== FILTER DONE ===")
    print("Input :", CSV_IN)
    print(f"Over  : {out_over_path} | rows={len(df_over)}")
    print(f"Under : {out_under_path} | rows={len(df_under)}")

    # mini-check
    if "fault2" in df.columns:
        print("\n=== fault2 distribution (overhang) ===")
        print(df_over["fault2"].astype("string").value_counts(dropna=False).to_dict())
        print("\n=== fault2 distribution (underhang) ===")
        print(df_under["fault2"].astype("string").value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()
