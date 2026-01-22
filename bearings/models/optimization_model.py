# - RF base (multiclass 0..5 con LabelEncoder fijo) 
# - Permutation importance con CV (ranking estable) 
# - Ablation TOP-K con CV 
# - Regla automática para elegir K óptimo (menor K dentro de best_mean - best_std) 
# - GridSearchCV sobre esas features 
#   1. n_estimators # Número de árboles del bosque; más árboles = modelo más estable pero más lento. 
#   2. max_depth # Profundidad máxima de cada árbol; limita la complejidad y evita sobreajuste. 
#   3. min_samples_leaf # Número mínimo de muestras en una hoja; fuerza reglas más generales. 
#   4. min_samples_split # Mínimo de muestras para dividir un nodo; evita divisiones poco fiables. 
# - Guarda modelo final y artefactos

import os
import time
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import joblib


# CONFIGURACION
CSV_PATH = "data/features_train.csv"

# Columnas que NO son features
ID_COL = "id"
LABEL_COL = "fault"
FILE_COL = "file"
EXTRA_COL = "fault2" # sublabel
META_COLS = [ID_COL, LABEL_COL, FILE_COL, EXTRA_COL]

SEED = 42
N_SPLITS = 5

CLASS_ORDER = [
    "horizontal-misalignment",
    "imbalance",
    "normal",
    "overhang",
    "underhang",
    "vertical-misalignment",
]

# K features a probar en ablación
TOP_K_LIST = [10, 20, 30, 40, 50, 75, 100, 125]

# RF base
RF_BASE = dict(
    n_estimators=200,
    random_state=SEED,
    class_weight="balanced_subsample",
    min_samples_leaf=2,
    n_jobs=-1,
)

# GridSearch
GRID = {
    "n_estimators": [200, 400, 800],
    "max_depth": [None, 10, 20, 30],
    "min_samples_leaf": [1, 2, 4],
    "min_samples_split": [2, 5, 10],
}

DIR_FEATURES_INFO = "models/final_model/features_info"
DIR_RESULTS = "models/final_model"

def clean_X(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.fillna(X.median(numeric_only=True))

def metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

def mean_std(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(x.std(ddof=0))


# MAIN:
def main():
    t0 = time.time()
    os.makedirs(DIR_FEATURES_INFO, exist_ok=True)
    os.makedirs(DIR_RESULTS, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    if LABEL_COL not in df.columns:
        raise ValueError(f"El CSV debe tener columna '{LABEL_COL}'.")

    df[LABEL_COL] = df[LABEL_COL].astype(str)

    # Meta cols presentes (por si alguna no existe en este CSV)
    meta_cols = [c for c in META_COLS if c in df.columns]

    # Features = todo menos meta
    feat_cols = [c for c in df.columns if c not in meta_cols]
    if not feat_cols:
        raise ValueError(f"No hay columnas de features. Meta detectadas: {meta_cols}")

    X_all = clean_X(df, feat_cols)

    # LabelEncoder
    le = LabelEncoder()
    le.fit(CLASS_ORDER)

    # Miramos que no haya etiquetas raras
    unknown = sorted(set(df[LABEL_COL].unique()) - set(CLASS_ORDER))
    if unknown:
        raise ValueError(f"Etiquetas fuera de CLASS_ORDER: {unknown}")

    y = le.transform(df[LABEL_COL].values)

    print("\n=== DATASET ===")
    print("CSV:", CSV_PATH)
    print("Rows:", len(df))
    print("Features:", len(feat_cols))
    print("Classes:", df[LABEL_COL].value_counts().to_dict())
    print("Meta cols removed:", meta_cols)
    print(f"Load time: {time.time() - t0:.2f}s")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    # BASE RF DEMO
    print("\n=== BASE RANDOM FOREST (1 fold demo) ===")
    tr_idx, te_idx = next(skf.split(X_all, y))
    X_tr, X_te = X_all.iloc[tr_idx], X_all.iloc[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    rf_demo = RandomForestClassifier(**RF_BASE)
    t_fit = time.time()
    rf_demo.fit(X_tr, y_tr)
    fit_s = time.time() - t_fit

    t_pred = time.time()
    pred = rf_demo.predict(X_te)
    pred_s = time.time() - t_pred

    print("Fit time (demo):", round(fit_s, 3), "s | Pred time (demo):", round(pred_s, 3), "s")
    print(metrics(y_te, pred))
    print("\nConfusion matrix (demo):")
    print(confusion_matrix(y_te, pred, labels=np.arange(len(CLASS_ORDER))))
    print("\nClassification report (demo):")
    print(classification_report(y_te, pred, target_names=CLASS_ORDER, zero_division=0))

    # Guardamos importancias rápidas del demo (pesos que el modelo le da a cada una)
    fast_df = pd.DataFrame({
        "feature": feat_cols,
        "importance": rf_demo.feature_importances_
    }).sort_values("importance", ascending=False)
    fast_path = os.path.join(DIR_FEATURES_INFO, "rf_feature_importances_fast_demo.csv")
    fast_df.to_csv(fast_path, index=False)
    print("\nSaved:", fast_path)

    # PERMUTATION IMPORTANCE con CV
    print("\n=== PERMUTATION IMPORTANCE (CV aggregated) ===")
    imp_folds = []
    fast_folds = []

    for fold, (tr, te) in enumerate(skf.split(X_all, y), start=1):
        rf = RandomForestClassifier(**RF_BASE)
        rf.fit(X_all.iloc[tr], y[tr])

        p = rf.predict(X_all.iloc[te])
        rec = recall_score(y[te], p, average="macro", zero_division=0)
        print(f"Fold {fold}/{N_SPLITS}: recall_macro={rec:.4f}")

        # importances fast
        fast_folds.append(rf.feature_importances_)

        # Permutation importance en el test fold
        perm = permutation_importance(
            rf,
            X_all.iloc[te],
            y[te],
            n_repeats=5,
            random_state=SEED,
            scoring="recall_macro",
            n_jobs=-1
        )
        imp_folds.append(perm.importances_mean)

    imp_folds = np.vstack(imp_folds) # (n_folds, n_features)
    fast_folds = np.vstack(fast_folds) # (n_folds, n_features)

    imp_mean = imp_folds.mean(axis=0)
    imp_std = imp_folds.std(axis=0, ddof=0)

    fast_mean = fast_folds.mean(axis=0)
    fast_std = fast_folds.std(axis=0, ddof=0)

    perm_df = pd.DataFrame({
        "feature": feat_cols,
        "importance_mean": imp_mean,
        "importance_std": imp_std,
        "fast_importance_mean": fast_mean,
        "fast_importance_std": fast_std,
    })

    # Ranking determinista de las importancias:
    # 1º mayor importance_mean
    # 2º menor importance_std
    # 3º mayor fast_importance_mean
    perm_df = perm_df.sort_values(
        by=["importance_mean", "importance_std", "fast_importance_mean", "feature"],
        ascending=[False, True, False, True],
        kind="mergesort"  # estable
    ).reset_index(drop=True)

    perm_path = os.path.join(DIR_FEATURES_INFO, "rf_feature_importances_permutation_cv.csv")
    perm_df.to_csv(perm_path, index=False)
    print("\nSaved:", perm_path)

    # ABLATION TOP-K (CV)
    print("\n=== ABLATION (TOP-K by permutation ranking, CV) ===")
    n_feat_total = len(feat_cols)
    top_k_list = [k for k in TOP_K_LIST if k <= n_feat_total]
    if not top_k_list:
        raise ValueError(f"TOP_K_LIST no tiene valores válidos (<= {n_feat_total}).")

    ablation_rows = []
    for k in top_k_list:
        top_feats = perm_df["feature"].head(k).tolist()
        Xk = clean_X(df, top_feats)

        scores = []
        for tr, te in skf.split(Xk, y):
            clf = RandomForestClassifier(**RF_BASE)
            clf.fit(Xk.iloc[tr], y[tr])
            p = clf.predict(Xk.iloc[te])
            scores.append(recall_score(y[te], p, average="macro", zero_division=0))

        m, s = mean_std(scores)
        ablation_rows.append({
            "top_k": int(k),
            "recall_macro_mean": m,
            "recall_macro_std": s,
        })
        print(f"TOP-{k}: recall_macro = {m:.4f} ± {s:.4f}")

    ablation_df = pd.DataFrame(ablation_rows).sort_values("top_k")
    ablation_path = os.path.join(DIR_RESULTS, "rf_ablation_results.csv")
    ablation_df.to_csv(ablation_path, index=False)
    print("\nSaved:", ablation_path)

    # Regla para elegir K óptimo
    # K* = menor K tal que mu_K >= mu_best - sigma_best
    best_row = ablation_df.loc[ablation_df["recall_macro_mean"].idxmax()]
    best_mean = float(best_row["recall_macro_mean"])
    best_std = float(best_row["recall_macro_std"])

    candidate = ablation_df[
        ablation_df["recall_macro_mean"] >= (best_mean - best_std)
    ].sort_values("top_k").iloc[0]

    best_k = int(candidate["top_k"])

    print(
        f"\nSelected TOP-K for GridSearch: {best_k} "
        f"(best={best_mean:.4f} ± {best_std:.4f}, "
        f"chosen={float(candidate['recall_macro_mean']):.4f} ± {float(candidate['recall_macro_std']):.4f})"
    )

    best_feats = perm_df["feature"].head(best_k).tolist()
    sel_feats_path = os.path.join(DIR_FEATURES_INFO, "rf_selected_features_for_gridsearch.csv")
    pd.Series(best_feats, name="feature").to_csv(sel_feats_path, index=False)
    print("Saved:", sel_feats_path)

    # GRIDSEARCH con CV con las K features:
    print("\n=== GRIDSEARCH (CV) ===")
    X_best = clean_X(df, best_feats)

    base_for_gs = RandomForestClassifier(
        random_state=SEED,
        class_weight="balanced_subsample",
        n_jobs=-1
    )

    gs = GridSearchCV(
        estimator=base_for_gs,
        param_grid=GRID,
        scoring="recall_macro",
        cv=skf,
        verbose=1,
        n_jobs=-1
    )

    t_gs = time.time()
    gs.fit(X_best, y)
    gs_s = time.time() - t_gs

    print("\nBest params:", gs.best_params_)
    print("Best CV recall_macro:", gs.best_score_)
    print("GridSearch time:", round(gs_s, 2), "s")

    gs_path = os.path.join(DIR_RESULTS, "rf_gridsearch_results.csv")
    pd.DataFrame(gs.cv_results_).to_csv(gs_path, index=False)
    print("\nSaved:", gs_path)

    # MODELO FINAL: métricas CV + confusión matrix
    print("\n=== FINAL MODEL (CV metrics + save) ===")
    final_rf = gs.best_estimator_

    cv_rows = []
    cms = []

    for fold, (tr, te) in enumerate(skf.split(X_best, y), start=1):
        clf = RandomForestClassifier(**final_rf.get_params())
        clf.fit(X_best.iloc[tr], y[tr])
        p = clf.predict(X_best.iloc[te])

        m = metrics(y[te], p)
        cv_rows.append(m)
        cms.append(confusion_matrix(y[te], p, labels=np.arange(len(CLASS_ORDER))))

        print(f"Fold {fold}: recall_macro={m['recall_macro']:.4f} | acc={m['accuracy']:.4f}")

    cv_df = pd.DataFrame(cv_rows)
    print("\n=== FINAL CV SUMMARY (mean ± std) ===")
    for met in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
        mu = float(cv_df[met].mean())
        sd = float(cv_df[met].std(ddof=0))
        print(f"{met}: {mu:.4f} ± {sd:.4f}")

    # Confusion matrix (suma de folds)
    cm_sum = np.sum(cms, axis=0)
    cm_df = pd.DataFrame(cm_sum, index=CLASS_ORDER, columns=CLASS_ORDER)

    print("\n=== CONFUSION MATRIX (SUM OVER CV FOLDS) ===")
    print(cm_df)

    final_rf.fit(X_best, y)
    bundle = {
        "model": final_rf,
        "features": best_feats,
        "label_encoder": le,
        "class_order": CLASS_ORDER,
        "meta_cols_removed": meta_cols,
        "csv_path": CSV_PATH,
    }
    out_pkl = os.path.join(DIR_RESULTS, "final_model.pkl")
    joblib.dump(bundle, out_pkl)

    print("\nSaved final model to:", out_pkl)
    print("\nDONE")


if __name__ == "__main__":
    main()
