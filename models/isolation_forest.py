import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score
)
import joblib
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.constants import (
    PROCESSED_DIR, METADATA_DIR, CLUSTERING_FEATURES
)
from utils.logger import get_logger

log = get_logger("isolation_forest")

MODEL_PATH = METADATA_DIR / "isolation_forest.pkl"

# Features to use — engineered features + cluster context
IF_FEATURES = CLUSTERING_FEATURES + [
    "auth_failure_rate", "privilege_abuse_score",
    "is_root_escalation", "anomalous_login_flag",
    "error_rate_combined", "service_diversity_score",
    "dst_host_suspicion_score", "same_service_ratio_delta",
    "file_access_intensity", "is_suspicious_file_activity",
    "bytes_ratio", "is_large_transfer",
    "peer_outlier_score", "bytes_above_peer_mean",
    "genai_composite_score", "api_call_pattern_score",
]


def load_data(split: str) -> tuple:
    path = PROCESSED_DIR / f"nslkdd_{split}_features.csv"
    df   = pd.read_csv(path)
    seen     = set()
    features = []
    for f in IF_FEATURES:
        if f in df.columns and f not in seen:
            features.append(f)
            seen.add(f)
    X = df[features].fillna(0)
    y = df["label_binary"]
    log.info(f"Loaded {split}: {X.shape[0]:,} rows, {X.shape[1]} features")
    return X, y, features


def train_isolation_forest(
    X_train: pd.DataFrame,
    contamination: float = 0.05,
    n_estimators: int = 100,
    random_state: int = 42,
) -> IsolationForest:
    """
    Train Isolation Forest.

    contamination = expected fraction of anomalies in training data.
    NSL-KDD train has ~46% attacks but we use 0.05 as conservative
    estimate matching real-world deployment assumptions.
    """
    log.info(
        f"Training Isolation Forest "
        f"(n_estimators={n_estimators}, contamination={contamination})..."
    )
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train)
    log.info("Training complete")
    return model


def evaluate(
    model: IsolationForest,
    X: pd.DataFrame,
    y: pd.Series,
    split: str = "test",
) -> dict:
    """
    Evaluate model. Isolation Forest outputs -1 (anomaly) or 1 (normal).
    We convert to binary: -1 -> 1 (attack), 1 -> 0 (normal).
    Also extract anomaly scores (lower = more anomalous).
    """
    raw_preds   = model.predict(X)
    y_pred      = np.where(raw_preds == -1, 1, 0)

    # Anomaly score: negative of decision function, normalised to [0,1]
    raw_scores  = model.decision_function(X)
    # More negative = more anomalous, flip and normalise
    scores      = -raw_scores
    scores      = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    f1          = f1_score(y, y_pred, zero_division=0)
    auc         = roc_auc_score(y, scores)
    fpr         = (
        ((y_pred == 1) & (y == 0)).sum() /
        ((y == 0).sum() + 1e-9)
    )

    log.info(f"\n--- Isolation Forest [{split}] ---")
    log.info(f"F1 Score : {f1:.4f}")
    log.info(f"ROC-AUC  : {auc:.4f}")
    log.info(f"FPR      : {fpr:.4f} ({fpr*100:.1f}% false positive rate)")
    log.info(f"\nClassification Report:\n"
             f"{classification_report(y, y_pred, target_names=['normal','attack'])}")
    log.info(f"Confusion Matrix:\n{confusion_matrix(y, y_pred)}")

    return {
        "f1"         : f1,
        "roc_auc"    : auc,
        "fpr"        : fpr,
        "y_pred"     : y_pred,
        "scores"     : scores,
    }


def run_isolation_forest() -> dict:
    """Master function — train, evaluate, save."""
    X_train, y_train, features = load_data("train")
    X_test,  y_test,  _        = load_data("test")

    # Keep only features present in both splits
    shared = [f for f in features if f in X_test.columns]
    X_train = X_train[shared]
    X_test  = X_test[shared]

    model   = train_isolation_forest(X_train)

    log.info("Evaluating on train split...")
    train_results = evaluate(model, X_train, y_train, "train")

    log.info("Evaluating on test split...")
    test_results  = evaluate(model, X_test,  y_test,  "test")

    # Save model
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": shared}, MODEL_PATH)
    log.info(f"Model saved -> {MODEL_PATH}")

    # Save scores to processed dir for downstream use
    test_df = pd.read_csv(PROCESSED_DIR / "nslkdd_test_features.csv")
    test_df["if_anomaly_score"] = test_results["scores"]
    test_df["if_prediction"]    = test_results["y_pred"]
    test_df.to_csv(
        PROCESSED_DIR / "nslkdd_test_if_scored.csv", index=False
    )
    log.info("Scored test set saved -> nslkdd_test_if_scored.csv")

    return test_results


if __name__ == "__main__":
    results = run_isolation_forest()
    print(f"\nFinal Test Results:")
    print(f"  F1      : {results['f1']:.4f}")
    print(f"  ROC-AUC : {results['roc_auc']:.4f}")
    print(f"  FPR     : {results['fpr']*100:.1f}%")