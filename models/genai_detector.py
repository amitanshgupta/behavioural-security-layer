import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score
)
import joblib
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.constants import PROCESSED_DIR, METADATA_DIR
from utils.logger import get_logger

log = get_logger("genai_detector")

MODEL_PATH = METADATA_DIR / "genai_detector.pkl"

# GenAI behavioural features — from genai_features.py
GENAI_FEATURES = [
    "api_call_pattern_score",
    "data_exfil_proxy",
    "rapid_request_flag",
    "genai_composite_score",
    "bytes_ratio",
    "src_bytes_peer_zscore",
    "bytes_above_peer_mean",
    "connection_rate",
    "is_large_transfer",
]


def load_data(split: str) -> tuple:
    path = PROCESSED_DIR / f"nslkdd_{split}_features.csv"
    df   = pd.read_csv(path)
    feats = [f for f in GENAI_FEATURES if f in df.columns]
    X = df[feats].fillna(0)
    y = df["label_binary"]
    log.info(f"Loaded {split}: {X.shape[0]:,} rows, {X.shape[1]} features")
    return X, y, feats


def train_genai_detector(
    X_train  : pd.DataFrame,
    y_train  : pd.Series,
    nu       : float = 0.05,
    kernel   : str   = "rbf",
) -> tuple:
    """
    Train One-Class SVM on NORMAL (benign) traffic only.
    The model learns what normal GenAI-adjacent behaviour looks like.
    Anything deviating from this is flagged as anomalous.

    nu = upper bound on fraction of outliers (0.05 = expect 5% anomalies)
    """
    # Train ONLY on normal samples — this is key for one-class learning
    X_normal = X_train[y_train == 0]
    log.info(
        f"Training One-Class SVM on {len(X_normal):,} normal samples "
        f"(nu={nu}, kernel={kernel})..."
    )

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_normal)

    model = OneClassSVM(nu=nu, kernel=kernel, gamma="scale")
    model.fit(X_scaled)

    log.info("Training complete")
    return model, scaler


def evaluate_genai(
    model  : OneClassSVM,
    scaler : StandardScaler,
    X      : pd.DataFrame,
    y      : pd.Series,
    split  : str = "test",
) -> dict:
    """
    One-Class SVM outputs +1 (normal) or -1 (anomaly).
    Convert: -1 -> 1 (attack), +1 -> 0 (normal).
    Decision function score = distance from boundary
    (more negative = more anomalous).
    """
    X_scaled = scaler.transform(X)
    raw_pred = model.predict(X_scaled)
    y_pred   = np.where(raw_pred == -1, 1, 0)

    # Anomaly score: flip decision function, normalise to [0,1]
    raw_scores = model.decision_function(X_scaled)
    scores     = -raw_scores
    scores     = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    f1  = f1_score(y, y_pred, zero_division=0)
    auc = roc_auc_score(y, scores)
    fpr = (
        ((y_pred == 1) & (y == 0)).sum() /
        ((y == 0).sum() + 1e-9)
    )

    log.info(f"\n--- GenAI Detector (One-Class SVM) [{split}] ---")
    log.info(f"F1      : {f1:.4f}")
    log.info(f"ROC-AUC : {auc:.4f}")
    log.info(f"FPR     : {fpr:.4f} ({fpr*100:.1f}%)")
    log.info(f"\n{classification_report(y, y_pred, target_names=['normal','attack'])}")
    log.info(f"Confusion Matrix:\n{confusion_matrix(y, y_pred)}")

    return {
        "f1"    : f1,
        "auc"   : auc,
        "fpr"   : fpr,
        "y_pred": y_pred,
        "scores": scores,
    }


def run_genai_detector() -> dict:
    """Master function — train, evaluate, save."""
    X_train, y_train, feats = load_data("train")
    X_test,  y_test,  _     = load_data("test")

    shared  = [f for f in feats if f in X_test.columns]
    X_train = X_train[shared]
    X_test  = X_test[shared]

    model, scaler = train_genai_detector(X_train, y_train)

    log.info("Evaluating on train split...")
    evaluate_genai(model, scaler, X_train, y_train, "train")

    log.info("Evaluating on test split...")
    results = evaluate_genai(model, scaler, X_test, y_test, "test")

    # Save
    joblib.dump(
        {"model": model, "scaler": scaler, "features": shared},
        MODEL_PATH
    )
    log.info(f"Model saved -> {MODEL_PATH}")

    # Save scores
    test_df = pd.read_csv(PROCESSED_DIR / "nslkdd_test_features.csv")
    test_df["genai_anomaly_score"] = results["scores"]
    test_df["genai_prediction"]    = results["y_pred"]
    test_df.to_csv(
        PROCESSED_DIR / "nslkdd_test_genai_scored.csv", index=False
    )
    log.info("Scored test set saved -> nslkdd_test_genai_scored.csv")

    return results


if __name__ == "__main__":
    results = run_genai_detector()
    print(f"\nFinal Test Results:")
    print(f"  F1      : {results['f1']:.4f}")
    print(f"  ROC-AUC : {results['auc']:.4f}")
    print(f"  FPR     : {results['fpr']*100:.1f}%")