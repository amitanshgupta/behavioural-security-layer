import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.ensemble import IsolationForest
import joblib
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.constants import PROCESSED_DIR, METADATA_DIR
from utils.logger import get_logger

log = get_logger("cert_model")

CERT_MODEL_PATH = METADATA_DIR / "cert_lightgbm.pkl"
CERT_IF_PATH    = METADATA_DIR / "cert_isolation_forest.pkl"

# Features for CERT model — excludes non-numeric identifiers
CERT_FEATURES = [
    # Logon signals
    "total_logons", "total_logoffs", "after_hours_logons",
    "weekend_logons", "unique_pcs", "logon_hour_mean", "logon_hour_std",
    # Device signals
    "usb_connects", "usb_after_hours", "usb_weekend", "unique_devices",
    # File signals
    "file_events", "executable_access",
    "after_hours_files", "unique_files",
    # HTTP signals
    "http_requests", "suspicious_domains", "job_site_visits",
    "cloud_storage_visits", "after_hours_http", "unique_domains",
    # Email signals
    "emails_sent", "external_emails", "high_risk_emails",
    "avg_email_size", "after_hours_emails",
    # User context
    "role_tier", "psych_risk_score", "O", "C", "E", "A", "N",
    # Peer deviation z-scores
    "total_logons_role_zscore", "after_hours_logons_role_zscore",
    "unique_pcs_role_zscore", "usb_connects_role_zscore",
    "exfil_attempts_role_zscore", "file_events_role_zscore",
    "http_requests_role_zscore", "suspicious_domains_role_zscore",
    "job_site_visits_role_zscore", "external_emails_role_zscore",
]

# Note: exfil_attempts excluded from features
# (it's used to build the label — would cause leakage)
# Note: exfil_attempts and its z-score excluded — used to build label
CERT_FEATURES = [f for f in CERT_FEATURES
                 if f not in ("exfil_attempts",
                              "exfil_attempts_role_zscore")]


def load_cert_data() -> tuple:
    """Load CERT feature matrix and split into train/test."""
    path = PROCESSED_DIR / "cert_features.csv"
    log.info(f"Loading {path}...")
    df = pd.read_csv(path, low_memory=False)

    features = [f for f in CERT_FEATURES if f in df.columns]
    X = df[features].fillna(0)
    y = df["label_binary"]

    log.info(f"CERT data: {len(df):,} rows, {len(features)} features")
    log.info(f"Label distribution: "
             f"{y.value_counts().to_dict()}")

    # Temporal split — use last 20% as test (respects time ordering)
    split_idx = int(len(df) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    log.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test, features


def train_cert_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val  : pd.DataFrame,
    y_val  : pd.Series,
) -> lgb.LGBMClassifier:
    """Train LightGBM on CERT user-day features."""
    log.info("Training CERT LightGBM...")

    model = lgb.LGBMClassifier(
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
    model.fit(
        X_train, y_train,
        eval_set  = [(X_val, y_val)],
        callbacks = [
            lgb.early_stopping(stopping_rounds=20, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    log.info(f"Training complete — best iteration: {model.best_iteration_}")
    return model


def train_cert_isolation_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> IsolationForest:
    """Train Isolation Forest on normal CERT behaviour only."""
    X_normal = X_train[y_train == 0]
    log.info(f"Training CERT Isolation Forest on "
             f"{len(X_normal):,} normal user-days...")
    model = IsolationForest(
        n_estimators  = 100,
        contamination = 0.07,   # ~7% attack rate in CERT
        random_state  = 42,
        n_jobs        = -1,
    )
    model.fit(X_normal)
    log.info("IF training complete")
    return model


def evaluate(
    model,
    X    : pd.DataFrame,
    y    : pd.Series,
    name : str,
    is_if: bool = False,
) -> dict:
    """Evaluate a model and log full metrics."""
    if is_if:
        raw    = model.predict(X)
        y_pred = np.where(raw == -1, 1, 0)
        scores = -model.decision_function(X)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    else:
        y_pred = model.predict(X)
        scores = model.predict_proba(X)[:, 1]

    f1  = f1_score(y, y_pred, zero_division=0)
    auc = roc_auc_score(y, scores)
    fpr = (
        ((y_pred == 1) & (y == 0)).sum() /
        ((y == 0).sum() + 1e-9)
    )
    prec = precision_score(y, y_pred, zero_division=0)
    rec  = recall_score(y, y_pred, zero_division=0)

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


def get_top_features(model, features: list, top_n: int = 15) -> None:
    """Log top features by importance."""
    imp = pd.Series(
        model.feature_importances_, index=features
    ).sort_values(ascending=False)

    log.info(f"\nTop {top_n} CERT features:")
    for feat, val in imp.head(top_n).items():
        log.info(f"  {feat:<45} {val:>8.1f}")


def run_cert_models() -> dict:
    """Master function — train and evaluate all CERT models."""
    X_train, X_test, y_train, y_test, features = load_cert_data()

    # Validation split from train
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=0.15, random_state=42, stratify=y_train
    )
    log.info(f"Train: {len(X_tr):,} | Val: {len(X_val):,} | "
             f"Test: {len(X_test):,}")

    # Train LightGBM
    lgb_model = train_cert_lightgbm(X_tr, y_tr, X_val, y_val)
    lgb_train = evaluate(lgb_model, X_train, y_train, "CERT LightGBM [train]")
    lgb_test  = evaluate(lgb_model, X_test,  y_test,  "CERT LightGBM [test]")
    get_top_features(lgb_model, features)

    # Train Isolation Forest
    if_model  = train_cert_isolation_forest(X_train, y_train)
    if_test   = evaluate(if_model, X_test, y_test,
                         "CERT IsolationForest [test]", is_if=True)

    # Save models
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": lgb_model, "features": features}, CERT_MODEL_PATH)
    joblib.dump({"model": if_model,  "features": features}, CERT_IF_PATH)
    log.info(f"Models saved -> {CERT_MODEL_PATH}, {CERT_IF_PATH}")

    # Save scored test set
    X_test_df = X_test.copy()
    X_test_df["label_binary"]   = y_test.values
    X_test_df["lgb_proba"]      = lgb_model.predict_proba(X_test)[:, 1]
    X_test_df["lgb_prediction"] = lgb_model.predict(X_test)

    raw_if = if_model.predict(X_test)
    X_test_df["if_prediction"]    = np.where(raw_if == -1, 1, 0)
    if_scores = -if_model.decision_function(X_test)
    X_test_df["if_anomaly_score"] = (
        (if_scores - if_scores.min()) /
        (if_scores.max() - if_scores.min() + 1e-9)
    )
    out = PROCESSED_DIR / "cert_test_scored.csv"
    X_test_df.to_csv(out, index=False)
    log.info(f"Scored test set saved -> {out}")

    return {
        "lgb_train": lgb_train,
        "lgb_test" : lgb_test,
        "if_test"  : if_test,
    }


if __name__ == "__main__":
    results = run_cert_models()

    print("\n" + "="*60)
    print("CERT MODEL RESULTS SUMMARY")
    print("="*60)
    for name, metrics in results.items():
        print(f"\n{metrics['model']}")
        print(f"  F1={metrics['f1']:.4f} | "
              f"AUC={metrics['roc_auc']:.4f} | "
              f"FPR={metrics['fpr']:.4f}")