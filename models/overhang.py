import os
import time
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import joblib

# CONFIGURACION
CSV_PATH = "data/features_overhang.csv"

ID_COL = "id"         
FILE_COL = "file"    
MAIN_LABEL_COL = "fault"  
SUB_LABEL_COL = "fault2"   

SEED = 42
N_SPLITS = 5

SUBCLASS_ORDER = ["ball_fault", "cage_fault", "outer_race"] # 0, 1, 2

# K features a probar en ablación
TOP_K_LIST = [10, 20, 30, 40, 50, 75, 100, 125]

RF_BASE = dict(
    n_estimators=200,
    random_state=SEED,
    class_weight="balanced_subsample",
    min_samples_leaf=2,
    n_jobs=-1,
)

GRID = {
    "n_estimators": [200, 400, 800],
    "max_depth": [None, 10, 20, 30],
    "min_samples_leaf": [1, 2, 4],
    "min_samples_split": [2, 5, 10],
}

OUT_DIR = "models/overhang"
DIR_FEATURES_INFO = os.path.join(OUT_DIR, "features_info")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DIR_FEATURES_INFO, exist_ok=True)

def clean_X(df, feat_cols):
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.fillna(X.median(numeric_only=True))


def metrics_typical(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def mean_std(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(x.std(ddof=0))

# MAIN
def main():
    t0 = time.time()
    df = pd.read_csv(CSV_PATH)

    # checks
    for c in [FILE_COL, SUB_LABEL_COL]:
        if c not in df.columns:
            raise ValueError(f"Falta columna obligatoria '{c}' en {CSV_PATH}.")

    if MAIN_LABEL_COL in df.columns:
        s = df[MAIN_LABEL_COL].astype(str).str.strip().str.lower()
        ok = s.isin(["3", "overhang"]).all()
        if not ok:
            raise ValueError("Este CSV debería venir filtrado SOLO a OVERHANG (fault='overhang' o 3).")

    # normalización de sublabels
    df[SUB_LABEL_COL] = df[SUB_LABEL_COL].astype(str).str.strip().str.lower()
    df[FILE_COL] = df[FILE_COL].astype(str)

    # filtramos por los sublabels válidos
    df = df[df[SUB_LABEL_COL].isin(SUBCLASS_ORDER)].copy()
    if len(df) == 0:
        raise ValueError("No hay filas con fault2 válido (ball_fault/cage_fault/outer_race).")

    drop_cols = {FILE_COL, SUB_LABEL_COL}
    if MAIN_LABEL_COL in df.columns:
        drop_cols.add(MAIN_LABEL_COL)
    if ID_COL in df.columns:
        drop_cols.add(ID_COL)

    feat_cols = [c for c in df.columns if c not in drop_cols]
    if not feat_cols:
        raise ValueError("No hay columnas de features tras quitar meta (id/fault/fault2/file).")

    X_all = clean_X(df, feat_cols)
    groups = df[FILE_COL].values

    le = LabelEncoder()
    le.fit(SUBCLASS_ORDER)
    y = le.transform(df[SUB_LABEL_COL].values)

    print("\n=== DATASET (OVERHANG SUBMODEL) ===")
    print("CSV:", CSV_PATH)
    print("Rows:", len(df))
    print("Features:", len(feat_cols))
    print("Subclasses:", df[SUB_LABEL_COL].value_counts().to_dict())
    print("Unique files:", pd.Series(groups).nunique())
    print(f"Load/prep time: {time.time() - t0:.2f}s")

    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    # BASE RF demo
    print("\n=== BASE RANDOM FOREST (1 fold demo) ===")
    tr_idx, te_idx = next(cv.split(X_all, y, groups=groups))
    X_tr, X_te = X_all.iloc[tr_idx], X_all.iloc[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    rf_demo = RandomForestClassifier(**RF_BASE)
    t1 = time.time()
    rf_demo.fit(X_tr, y_tr)
    print(f"Train time (demo): {time.time() - t1:.2f}s")

    pred_demo = rf_demo.predict(X_te)
    print(metrics_typical(y_te, pred_demo))
    print("\nConfusion matrix (demo):")
    print(confusion_matrix(y_te, pred_demo))
    print("\nClassification report (demo):")
    print(classification_report(y_te, pred_demo, target_names=SUBCLASS_ORDER, zero_division=0))

    fast_df = pd.DataFrame({
        "feature": feat_cols,
        "importance": rf_demo.feature_importances_,
    }).sort_values("importance", ascending=False)
    fast_demo_path = os.path.join(DIR_FEATURES_INFO, "rf_feature_importances_fast_demo.csv")
    fast_df.to_csv(fast_demo_path, index=False)
    print("Saved:", fast_demo_path)

    # PERMUTATION IMPORTANCE con CV
    print("\n=== PERMUTATION IMPORTANCE ===")
    imp_folds = []
    fast_folds = []

    for fold, (tr, te) in enumerate(cv.split(X_all, y, groups=groups), start=1):
        rf = RandomForestClassifier(**RF_BASE)
        rf.fit(X_all.iloc[tr], y[tr])

        # fold metric
        p = rf.predict(X_all.iloc[te])
        rec = recall_score(y[te], p, average="macro", zero_division=0)
        print(f"Fold {fold}/{N_SPLITS}: recall_macro={rec:.4f}")

        # fast importance del fold
        fast_folds.append(rf.feature_importances_)

        # permutation importance en test fold
        perm = permutation_importance(
            rf,
            X_all.iloc[te],
            y[te],
            n_repeats=5,
            random_state=SEED,
            scoring="recall_macro",
            n_jobs=-1,
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
        kind="mergesort"
    ).reset_index(drop=True)

    perm_path = os.path.join(DIR_FEATURES_INFO, "rf_feature_importances_permutation_cv.csv")
    perm_df.to_csv(perm_path, index=False)
    print("Saved:", perm_path)

    # ABLATION TOP-K (CV)
    print("\n=== ABLATION (TOP-K by ranking, CV) ===")
    n_feat_total = len(feat_cols)
    top_k_list = [k for k in TOP_K_LIST if k <= n_feat_total]
    if not top_k_list:
        raise ValueError(f"TOP_K_LIST no tiene valores válidos (<= {n_feat_total}).")

    ablation_rows = []
    for k in top_k_list:
        top_feats = perm_df["feature"].head(k).tolist()
        Xk = clean_X(df, top_feats)

        scores = []
        for tr, te in cv.split(Xk, y, groups=groups):
            clf = RandomForestClassifier(**RF_BASE)
            clf.fit(Xk.iloc[tr], y[tr])
            p = clf.predict(Xk.iloc[te])
            scores.append(recall_score(y[te], p, average="macro", zero_division=0))

        m, s = mean_std(scores)
        ablation_rows.append({"top_k": int(k), "recall_macro_mean": m, "recall_macro_std": s})
        print(f"TOP-{k}: recall_macro = {m:.4f} ± {s:.4f}")

    ablation_df = pd.DataFrame(ablation_rows).sort_values("top_k")
    ablation_path = os.path.join(OUT_DIR, "rf_ablation_results.csv")
    ablation_df.to_csv(ablation_path, index=False)
    print("Saved:", ablation_path)

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
    sel_path = os.path.join(DIR_FEATURES_INFO, "rf_selected_features_for_gridsearch.csv")
    pd.Series(best_feats, name="feature").to_csv(sel_path, index=False)
    print("Saved:", sel_path)

    # GRIDSEARCH CV con las K features:
    print("\n=== GRIDSEARCH (scoring=recall_macro) ===")
    X_best = clean_X(df, best_feats)

    gs = GridSearchCV(
        estimator=RandomForestClassifier(random_state=SEED, class_weight="balanced_subsample", n_jobs=-1),
        param_grid=GRID,
        scoring="recall_macro",
        cv=cv.split(X_best, y, groups=groups),
        verbose=1,
        n_jobs=-1,
    )

    tgs = time.time()
    gs.fit(X_best, y)
    print(f"GridSearch time: {time.time() - tgs:.2f}s")
    print("Best params:", gs.best_params_)
    print("Best CV recall_macro:", gs.best_score_)

    pd.DataFrame(gs.cv_results_).to_csv(os.path.join(OUT_DIR, "rf_gridsearch_results.csv"), index=False)

    # MODELO FINAL: métricas CV + confusión matrix
    print("\n=== FINAL MODEL ===")
    final_rf = gs.best_estimator_

    cv_rows = []
    cms = []
    for fold, (tr, te) in enumerate(cv.split(X_best, y, groups=groups), start=1):
        clf = RandomForestClassifier(**final_rf.get_params())
        clf.fit(X_best.iloc[tr], y[tr])
        p = clf.predict(X_best.iloc[te])

        m = metrics_typical(y[te], p)
        cv_rows.append(m)
        cms.append(confusion_matrix(y[te], p, labels=np.arange(len(SUBCLASS_ORDER))))
        print(f"Fold {fold}: recall_macro={m['recall_macro']:.4f} | acc={m['accuracy']:.4f}")

    cv_df = pd.DataFrame(cv_rows)
    print("\n=== FINAL CV SUMMARY (mean ± std) ===")
    for met in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
        mu = float(cv_df[met].mean())
        sd = float(cv_df[met].std(ddof=0))
        print(f"{met}: {mu:.4f} ± {sd:.4f}")

    # Confusion matrix (suma de folds)
    cm_sum = np.sum(cms, axis=0)
    cm_df = pd.DataFrame(cm_sum, index=SUBCLASS_ORDER, columns=SUBCLASS_ORDER)
    print("\n=== CONFUSION MATRIX (SUM OVER CV FOLDS) ===")
    print(cm_df)

    final_rf.fit(X_best, y)

    bundle = {
        "model": final_rf,
        "features": best_feats,
        "label_encoder": le,
        "subclass_order": SUBCLASS_ORDER,
        "parent": "overhang",
    }
    out_model = os.path.join(OUT_DIR, "overhang_model.pkl")
    joblib.dump(bundle, out_model)
    print("\nSaved:", out_model)
    print("\nDONE")


if __name__ == "__main__":
    main()
