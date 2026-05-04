import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.ensemble import IsolationForest
import lightgbm as lgb
import joblib
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.constants import PROCESSED_DIR, METADATA_DIR
from utils.logger import get_logger

log = get_logger("network_ids_model")

MODEL_PATH    = METADATA_DIR / "network_ids_lgb.pkl"
IF_MODEL_PATH = METADATA_DIR / "network_ids_if.pkl"


def load_unified() -> tuple:
    """Load unified NSL-KDD + CICIDS2017 dataset."""
    path = PROCESSED_DIR / "unified_network_features.csv"
    log.info(f"Loading {path}...")
    df = pd.read_csv(path, low_memory=False)

    feature_cols = [c for c in df.columns
                    if c not in ["label_binary", "dataset"]]
    X = df[feature_cols].fillna(0)
    y = df["label_binary"]
    source = df["dataset"]

    log.info(f"Loaded: {len(df):,} rows | "
             f"{len(feature_cols)} features | "
             f"attack rate: {y.mean()*100:.1f}%")
    return X, y, feature_cols, source


def evaluate(
    model,
    X        : pd.DataFrame,
    y        : pd.Series,
    name     : str,
    is_if    : bool = False,
    threshold: float = 0.5,
) -> dict:
    if is_if:
        raw    = model.predict(X)
        y_pred = np.where(raw == -1, 1, 0)
        scores = -model.decision_function(X)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    else:
        scores = model.predict_proba(X)[:, 1]
        y_pred = (scores >= threshold).astype(int)

    f1   = f1_score(y, y_pred, zero_division=0)
    prec = precision_score(y, y_pred, zero_division=0)
    rec  = recall_score(y, y_pred, zero_division=0)
    auc  = roc_auc_score(y, scores)
    fpr  = (
        ((y_pred == 1) & (y == 0)).sum() /
        ((y == 0).sum() + 1e-9)
    )

    log.info(f"\n--- {name} ---")
    log.info(f"F1={f1:.4f} | Precision={prec:.4f} | "
             f"Recall={rec:.4f} | AUC={auc:.4f} | FPR={fpr:.4f}")
    log.info(f"\n{classification_report(y, y_pred, target_names=['normal','attack'])}")
    log.info(f"Confusion Matrix:\n{confusion_matrix(y, y_pred)}")

    return {
        "model"    : name,
        "f1"       : round(f1, 4),
        "precision": round(prec, 4),
        "recall"   : round(rec, 4),
        "roc_auc"  : round(auc, 4),
        "fpr"      : round(fpr, 4),
    }


def evaluate_by_source(
    model   : lgb.LGBMClassifier,
    X       : pd.DataFrame,
    y       : pd.Series,
    source  : pd.Series,
) -> None:
    """Evaluate separately on NSL-KDD and CICIDS subsets."""
    for src in ["nslkdd", "cicids"]:
        mask   = source == src
        if mask.sum() == 0:
            continue
        X_sub  = X[mask]
        y_sub  = y[mask]
        y_pred = model.predict(X_sub)
        f1     = f1_score(y_sub, y_pred, zero_division=0)
        auc    = roc_auc_score(
            y_sub, model.predict_proba(X_sub)[:, 1]
        )
        fpr    = (
            ((y_pred == 1) & (y_sub == 0)).sum() /
            ((y_sub == 0).sum() + 1e-9)
        )
        log.info(f"  {src.upper():<10} F1={f1:.4f} | "
                 f"AUC={auc:.4f} | FPR={fpr:.4f}")


def run_network_ids() -> dict:
    """Master function — train and evaluate network IDS model."""
    X, y, features, source = load_unified()

    # Stratified train/test split preserving dataset source
    X_train, X_test, y_train, y_test, src_train, src_test = \
        train_test_split(
            X, y, source,
            test_size=0.20, random_state=42, stratify=y
        )

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=0.15, random_state=42, stratify=y_train
    )

    log.info(f"Train: {len(X_tr):,} | Val: {len(X_val):,} | "
             f"Test: {len(X_test):,}")

    # Train LightGBM
    log.info("Training Network IDS LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators     = 200,
        max_depth        = 7,
        learning_rate    = 0.05,
        num_leaves       = 63,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_samples= 20,
        class_weight     = "balanced",
        random_state     = 42,
        n_jobs           = -1,
        verbose          = -1,
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set  = [(X_val, y_val)],
        callbacks = [
            lgb.early_stopping(stopping_rounds=20, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    log.info(f"LGB training complete — "
             f"best iteration: {lgb_model.best_iteration_}")

    # Train Isolation Forest on normal traffic only
    log.info("Training Network IDS Isolation Forest...")
    X_normal = X_train[y_train == 0]
    if_model = IsolationForest(
        n_estimators  = 100,
        contamination = 0.24,
        random_state  = 42,
        n_jobs        = -1,
    )
    if_model.fit(X_normal)

    # Evaluate
    lgb_results = evaluate(lgb_model, X_test, y_test,
                           "Network IDS LightGBM [test]")
    if_results  = evaluate(if_model, X_test, y_test,
                           "Network IDS IF [test]", is_if=True)

    # Per-dataset breakdown
    log.info("\nPer-dataset breakdown (LightGBM):")
    evaluate_by_source(lgb_model, X_test, y_test, src_test)

    # Feature importance
    imp = pd.Series(
        lgb_model.feature_importances_, index=features
    ).sort_values(ascending=False)
    log.info(f"\nTop 15 features:")
    for feat, val in imp.head(15).items():
        log.info(f"  {feat:<45} {val:>8.1f}")

    # Save
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": lgb_model, "features": features}, MODEL_PATH)
    joblib.dump({"model": if_model,  "features": features}, IF_MODEL_PATH)
    log.info(f"Models saved -> {MODEL_PATH}, {IF_MODEL_PATH}")

    # Save scored test
    out_df = X_test.copy()
    out_df["label_binary"]   = y_test.values
    out_df["dataset"]        = src_test.values
    out_df["lgb_proba"]      = lgb_model.predict_proba(X_test)[:, 1]
    out_df["lgb_prediction"] = lgb_model.predict(X_test)
    raw_if = if_model.predict(X_test)
    out_df["if_prediction"]  = np.where(raw_if == -1, 1, 0)
    out_df.to_csv(
        PROCESSED_DIR / "network_ids_test_scored.csv", index=False
    )
    log.info("Scored test saved -> network_ids_test_scored.csv")

    return {"lgb": lgb_results, "if": if_results}


if __name__ == "__main__":
    results = run_network_ids()

    print(f"\n{'='*60}")
    print(f"NETWORK IDS MODEL RESULTS")
    print(f"{'='*60}")
    for name, m in results.items():
        print(f"\n{m['model']}")
        print(f"  F1={m['f1']:.4f} | AUC={m['roc_auc']:.4f} | "
              f"FPR={m['fpr']*100:.1f}%")