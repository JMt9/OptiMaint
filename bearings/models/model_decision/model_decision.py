import time
import numpy as np
import pandas as pd

from prettytable import PrettyTable

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

# CONFIGURACIÓN:
CSV_PATH = "data/features_train.csv" 
ID_COL = "id"   
LABEL_COL = "fault"
FILE_COL = "file"
EXTRA = "fault2"        
META_COLS = [ID_COL, LABEL_COL, FILE_COL, EXTRA] # todo lo que NO son features
N_SPLITS = 5
SEED = 42

# Orden fijo para el labelencoder
CLASS_ORDER = [
    "horizontal-misalignment", # 0
    "imbalance", # 1
    "normal", # 2
    "overhang", # 3
    "underhang", # 4
    "vertical-misalignment", # 5
]


def _mean_std_str(values):
    v = np.asarray(values, dtype=float)
    return f"{v.mean():.4f} ± {v.std(ddof=0):.4f}"


def main():
    t0 = time.time()
    df = pd.read_csv(CSV_PATH)

    if LABEL_COL not in df.columns:
        raise ValueError(f"El CSV debe tener columna '{LABEL_COL}'.")
    if FILE_COL not in df.columns:
        raise ValueError(f"El CSV debe tener columna '{FILE_COL}' (lo usarás luego en el pipeline).")

    df[LABEL_COL] = df[LABEL_COL].astype(str)
    df[FILE_COL] = df[FILE_COL].astype(str)
    df = df.dropna(subset=[LABEL_COL, FILE_COL]).copy()

    # Features = todas las columnas excepto meta
    feat_cols = [c for c in df.columns if c not in META_COLS]
    if not feat_cols:
        raise ValueError("No hay columnas de features (solo meta/label).")

    # X numérica
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    # Label encoding para categóricas
    y_str = df[LABEL_COL].values
    le = LabelEncoder()
    le.fit(CLASS_ORDER)

    unknown = set(pd.unique(y_str)) - set(CLASS_ORDER)
    if unknown:
        raise ValueError(
            f"Hay labels en el CSV que no están en CLASS_ORDER: {sorted(unknown)}"
        )

    y = le.transform(y_str)

    print("\n=== DATASET INFO ===")
    print("CSV:", CSV_PATH)
    print("Rows:", len(df))
    print("Features:", len(feat_cols))
    print("Classes present:", dict(pd.Series(y_str).value_counts()))
    print(f"Load/prep time: {time.time() - t0:.2f}s")

    # MODELOS:
    # 1 Baseline que predice la clase más frecuente 
    # 2 KNN
    # 3 Logistic Regression 
    # 4 RandomForest
    # 5 Más potente: XGBoost

    models = {
        "baseline_mostfreq": DummyClassifier(strategy="most_frequent"),

        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                multi_class="multinomial",
                solver="lbfgs",
                max_iter=4000,
                class_weight="balanced",
                random_state=SEED
            ))
        ]),

        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(
                n_neighbors=11,
                weights="distance"
            ))
        ]),

        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=SEED,
            class_weight="balanced_subsample",
            min_samples_leaf=2
        ),

        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=SEED
        ),
    }

    # STRATIFIED K-FOLD 
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    results = []

    for name, model in models.items():
        accs, precs, recs, f1ms = [], [], [], []
        train_ts = []

        for tr_idx, te_idx in skf.split(X, y):
            X_train, X_test = X.iloc[tr_idx], X.iloc[te_idx]
            y_train, y_test = y[tr_idx], y[te_idx]

            fit_kwargs = {}

            # Para XGB: sample_weight balanceado
            if name == "xgboost":
                y_ser = pd.Series(y_train)
                counts = y_ser.value_counts()
                N = len(y_ser)
                K = len(counts)
                w = {c: N / (K * cnt) for c, cnt in counts.items()}
                fit_kwargs["sample_weight"] = np.array([w[c] for c in y_train])

            t1 = time.time()
            model.fit(X_train, y_train, **fit_kwargs)
            train_ts.append(time.time() - t1)

            pred = model.predict(X_test)

            # Métricas básicas usando macro
            accs.append(accuracy_score(y_test, pred))
            precs.append(precision_score(y_test, pred, average="macro", zero_division=0))
            recs.append(recall_score(y_test, pred, average="macro", zero_division=0))
            f1ms.append(f1_score(y_test, pred, average="macro", zero_division=0))

        results.append({
            "model": name,
            "accuracy": _mean_std_str(accs),
            "precision_macro": _mean_std_str(precs),
            "recall_macro": _mean_std_str(recs),
            "f1_macro": _mean_std_str(f1ms),
            "train_s": _mean_std_str(train_ts),
        })

    def parse_mean(ms):
        return float(ms.split("±")[0].strip())
    results = sorted(results, key=lambda r: parse_mean(r["f1_macro"]), reverse=True)

    # Tabla resultados:
    table = PrettyTable()
    table.field_names = [
        "model",
        "acc",
        "prec_macro",
        "recall_macro",
        "f1_macro",
        "train_s",
    ]

    for r in results:
        table.add_row([
            r["model"],
            r["accuracy"],
            r["precision_macro"],
            r["recall_macro"],
            r["f1_macro"],
            r["train_s"],
        ])

    print("\n=== BENCHMARK (StratifiedKFold mean ± std) ===")
    print(table)

    out_df = pd.DataFrame(results)
    out_path = "models/model_decision/benchmark_results_pretty.csv"
    out_df.to_csv(out_path, index=False)
    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()
