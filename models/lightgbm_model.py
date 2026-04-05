import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score, recall_score
)
from sklearn.model_selection import StratifiedKFold
import joblib
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.constants import (
    PROCESSED_DIR, METADATA_DIR, CLUSTERING_FEATURES
)
from utils.logger import get_logger

log = get_logger("lightgbm_model")

MODEL_PATH = METADATA_DIR / "lightgbm_model.pkl"

# Full feature set for LightGBM — uses everything
LGB_FEATURES = CLUSTERING_FEATURES + [
    "auth_failure_rate", "privilege_abuse_score",
    "is_root_escalation", "anomalous_login_flag",
    "error_rate_combined", "service_diversity_score",
    "dst_host_suspicion_score", "same_service_ratio_delta",
    "file_access_intensity", "is_suspicious_file_activity",
    "bytes_ratio", "is_large_transfer", "connection_rate",
    "peer_outlier_score", "bytes_above_peer_mean",
    "src_bytes_peer_zscore", "dst_bytes_peer_zscore",
    "count_peer_zscore", "duration_peer_zscore",
    "genai_composite_score", "api_call_pattern_score",
    "data_exfil_proxy", "rapid_request_flag",
]


def load_data(split: str) -> tuple:
    path = PROCESSED_DIR / f"nslkdd_{split}_features.csv"
    df   = pd.read_csv(path)
    
    # Deduplicate feature list before selecting columns
    seen     = set()
    features = []
    for f in LGB_FEATURES:
        if f in df.columns and f not in seen:
            features.append(f)
            seen.add(f)

    X = df[features].fillna(0)
    y = df["label_binary"]
    log.info(f"Loaded {split}: {X.shape[0]:,} rows, {X.shape[1]} features")
    return X, y, features


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val:   pd.DataFrame,
    y_val:   pd.Series,
) -> lgb.LGBMClassifier:
    """
    Train LightGBM with early stopping on validation set.
    Parameters from proposal: 200 estimators, max_depth=7, lr=0.05
    """
    log.info("Training LightGBM classifier...")

    model = lgb.LGBMClassifier(
        n_estimators      = 200,
        max_depth         = 7,
        learning_rate     = 0.05,
        num_leaves        = 63,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        min_child_samples = 20,
        class_weight      = "balanced",
        random_state      = 42,
        n_jobs            = -1,
        verbose           = -1,
    )

    model.fit(
        X_train, y_train,
        eval_set              = [(X_val, y_val)],
        callbacks             = [
            lgb.early_stopping(stopping_rounds=20, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    log.info(f"Training complete — best iteration: {model.best_iteration_}")
    return model


def evaluate(
    model    : lgb.LGBMClassifier,
    X        : pd.DataFrame,
    y        : pd.Series,
    split    : str = "test",
    threshold: float = 0.5,
) -> dict:
    """
    Full evaluation with adjustable decision threshold.
    Returns predictions, probabilities and all metrics.
    """
    proba  = model.predict_proba(X)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    f1        = f1_score(y, y_pred, zero_division=0)
    precision = precision_score(y, y_pred, zero_division=0)
    recall    = recall_score(y, y_pred, zero_division=0)
    auc       = roc_auc_score(y, proba)
    fpr       = (
        ((y_pred == 1) & (y == 0)).sum() /
        ((y == 0).sum() + 1e-9)
    )

    log.info(f"\n--- LightGBM [{split}] (threshold={threshold}) ---")
    log.info(f"F1        : {f1:.4f}")
    log.info(f"Precision : {precision:.4f}")
    log.info(f"Recall    : {recall:.4f}")
    log.info(f"ROC-AUC   : {auc:.4f}")
    log.info(f"FPR       : {fpr:.4f} ({fpr*100:.1f}%)")
    log.info(f"\nClassification Report:\n"
             f"{classification_report(y, y_pred, target_names=['normal','attack'])}")
    log.info(f"Confusion Matrix:\n{confusion_matrix(y, y_pred)}")

    return {
        "f1"       : f1,
        "precision": precision,
        "recall"   : recall,
        "roc_auc"  : auc,
        "fpr"      : fpr,
        "y_pred"   : y_pred,
        "proba"    : proba,
    }


def get_feature_importance(
    model   : lgb.LGBMClassifier,
    features: list,
    top_n   : int = 20,
) -> pd.DataFrame:
    """Return top_n features by importance."""
    imp = pd.DataFrame({
        "feature"   : features,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    log.info(f"\nTop {top_n} features:")
    for _, row in imp.head(top_n).iterrows():
        log.info(f"  {row['feature']:<40} {row['importance']:>8.1f}")

    return imp


def run_lightgbm() -> dict:
    """Master function — train, evaluate, save."""
    X_train, y_train, features = load_data("train")
    X_test,  y_test,  _        = load_data("test")

    # Deduplicate while preserving order
    seen   = set()
    shared = []
    for f in features:
        if f in X_test.columns and f not in seen:
            shared.append(f)
            seen.add(f)

    X_train = X_train[shared]
    X_test  = X_test[shared]

    # Use 15% of train as validation for early stopping
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size    = 0.15,
        random_state = 42,
        stratify     = y_train,
    )
    log.info(f"Train: {len(X_tr):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    model = train_lightgbm(X_tr, y_tr, X_val, y_val)

    log.info("Evaluating on train split...")
    train_results = evaluate(model, X_train, y_train, "train")

    log.info("Evaluating on test split...")
    test_results  = evaluate(model, X_test, y_test, "test")

    # Feature importance
    imp_df = get_feature_importance(model, shared)
    imp_df.to_csv(METADATA_DIR / "lgb_feature_importance.csv", index=False)

    # Save model
    joblib.dump({"model": model, "features": shared}, MODEL_PATH)
    log.info(f"Model saved -> {MODEL_PATH}")

    # Save scored test set
    test_df = pd.read_csv(PROCESSED_DIR / "nslkdd_test_features.csv")
    test_df["lgb_proba"]      = test_results["proba"]
    test_df["lgb_prediction"] = test_results["y_pred"]
    test_df.to_csv(
        PROCESSED_DIR / "nslkdd_test_lgb_scored.csv", index=False
    )
    log.info("Scored test set saved -> nslkdd_test_lgb_scored.csv")

    return test_results


if __name__ == "__main__":
    results = run_lightgbm()
    print(f"\nFinal Test Results:")
    print(f"  F1        : {results['f1']:.4f}")
    print(f"  Precision : {results['precision']:.4f}")
    print(f"  Recall    : {results['recall']:.4f}")
    print(f"  ROC-AUC   : {results['roc_auc']:.4f}")
    print(f"  FPR       : {results['fpr']*100:.1f}%")