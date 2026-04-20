import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import lightgbm as lgb
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.constants import PROCESSED_DIR, METADATA_DIR
from utils.logger import get_logger
from evaluation.metrics import METRICS_DIR

log = get_logger("ablation")

# Feature groups matching each novel contribution
FEATURE_GROUPS = {
    "baseline_raw": [
        "duration", "src_bytes", "dst_bytes", "count", "srv_count",
        "serror_rate", "rerror_rate", "same_srv_rate", "diff_srv_rate",
        "dst_host_count", "dst_host_srv_count",
    ],
    "peer_features": [
        "src_bytes_peer_zscore", "dst_bytes_peer_zscore",
        "count_peer_zscore", "duration_peer_zscore",
        "peer_outlier_score", "bytes_above_peer_mean",
    ],
    "auth_features": [
        "auth_failure_rate", "privilege_abuse_score",
        "is_root_escalation", "anomalous_login_flag",
    ],
    "session_features": [
        "error_rate_combined", "service_diversity_score",
        "dst_host_suspicion_score", "same_service_ratio_delta",
    ],
    "genai_features": [
        "genai_composite_score", "api_call_pattern_score",
        "data_exfil_proxy", "rapid_request_flag",
    ],
    "network_features": [
        "bytes_ratio", "is_large_transfer", "connection_rate",
    ],
}


def train_and_eval(
    X_train : pd.DataFrame,
    y_train : pd.Series,
    X_test  : pd.DataFrame,
    y_test  : pd.Series,
    label   : str,
) -> dict:
    """Train a fresh LightGBM and evaluate on test."""
    model = lgb.LGBMClassifier(
        n_estimators  = 100,
        max_depth     = 7,
        learning_rate = 0.05,
        class_weight  = "balanced",
        random_state  = 42,
        n_jobs        = -1,
        verbose       = -1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    proba  = model.predict_proba(X_test)[:, 1]

    f1  = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, proba)
    fpr = (
        ((y_pred == 1) & (y_test == 0)).sum() /
        ((y_test == 0).sum() + 1e-9)
    )

    log.info(f"  {label:<45} F1={f1:.4f} AUC={auc:.4f} FPR={fpr:.4f}")
    return {
        "configuration": label,
        "n_features"   : X_train.shape[1],
        "f1"           : round(float(f1), 4),
        "roc_auc"      : round(float(auc), 4),
        "fpr"          : round(float(fpr), 4),
    }


def run_ablation() -> pd.DataFrame:
    """
    Ablation study — measures contribution of each feature group.

    Configurations tested:
      1. Baseline only (raw NSL-KDD features)
      2. + Peer features (Novel Contribution 1)
      3. + Auth features
      4. + Session features
      5. + Network engineered features
      6. + GenAI features (Novel Contribution 4)
      7. Full model (all features)
      8. Leave-one-out for each group
    """
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(PROCESSED_DIR / "nslkdd_train_features.csv")
    test_df  = pd.read_csv(PROCESSED_DIR / "nslkdd_test_features.csv")

    y_train  = train_df["label_binary"]
    y_test   = test_df["label_binary"]

    results  = []
    log.info("Running ablation study...")
    log.info(f"{'Configuration':<45} {'F1':>6} {'AUC':>7} {'FPR':>7}")
    log.info("-" * 70)

    # Helper — get valid features from a list
    def valid(feats):
        return [f for f in feats
                if f in train_df.columns and f in test_df.columns]

    # 1. Baseline only
    base_feats = valid(FEATURE_GROUPS["baseline_raw"])
    results.append(train_and_eval(
        train_df[base_feats], y_train,
        test_df[base_feats],  y_test,
        "1. Baseline (raw features only)"
    ))

    # 2. + Peer features
    feats = valid(
        FEATURE_GROUPS["baseline_raw"] +
        FEATURE_GROUPS["peer_features"]
    )
    results.append(train_and_eval(
        train_df[feats], y_train,
        test_df[feats],  y_test,
        "2. + Peer features (NC1)"
    ))

    # 3. + Auth features
    feats = valid(
        FEATURE_GROUPS["baseline_raw"] +
        FEATURE_GROUPS["peer_features"] +
        FEATURE_GROUPS["auth_features"]
    )
    results.append(train_and_eval(
        train_df[feats], y_train,
        test_df[feats],  y_test,
        "3. + Auth features"
    ))

    # 4. + Session features
    feats = valid(
        FEATURE_GROUPS["baseline_raw"] +
        FEATURE_GROUPS["peer_features"] +
        FEATURE_GROUPS["auth_features"] +
        FEATURE_GROUPS["session_features"]
    )
    results.append(train_and_eval(
        train_df[feats], y_train,
        test_df[feats],  y_test,
        "4. + Session features"
    ))

    # 5. + Network engineered
    feats = valid(
        FEATURE_GROUPS["baseline_raw"] +
        FEATURE_GROUPS["peer_features"] +
        FEATURE_GROUPS["auth_features"] +
        FEATURE_GROUPS["session_features"] +
        FEATURE_GROUPS["network_features"]
    )
    results.append(train_and_eval(
        train_df[feats], y_train,
        test_df[feats],  y_test,
        "5. + Network engineered features"
    ))

    # 6. Full model
    lgb_data   = joblib.load(METADATA_DIR / "lightgbm_model.pkl")
    all_feats  = valid(lgb_data["features"])
    results.append(train_and_eval(
        train_df[all_feats], y_train,
        test_df[all_feats],  y_test,
        "6. Full model (all features)"
    ))

    # 7. + GenAI only added to baseline
    feats = valid(
        FEATURE_GROUPS["baseline_raw"] +
        FEATURE_GROUPS["genai_features"]
    )
    results.append(train_and_eval(
        train_df[feats], y_train,
        test_df[feats],  y_test,
        "7. Baseline + GenAI features only (NC4)"
    ))

    # 8. Leave-one-out — remove peer features
    feats = valid([
        f for f in all_feats
        if f not in FEATURE_GROUPS["peer_features"]
    ])
    results.append(train_and_eval(
        train_df[feats], y_train,
        test_df[feats],  y_test,
        "8. Full - Peer features (NC1 impact)"
    ))

    # 9. Leave-one-out — remove GenAI features
    feats = valid([
        f for f in all_feats
        if f not in FEATURE_GROUPS["genai_features"]
    ])
    results.append(train_and_eval(
        train_df[feats], y_train,
        test_df[feats],  y_test,
        "9. Full - GenAI features (NC4 impact)"
    ))

    ablation_df = pd.DataFrame(results)

    # Save
    out = METRICS_DIR / "ablation_study.csv"
    ablation_df.to_csv(out, index=False)
    log.info(f"\nAblation study saved -> {out}")

    return ablation_df


if __name__ == "__main__":
    df = run_ablation()

    print("\n" + "="*70)
    print("ABLATION STUDY RESULTS")
    print("="*70)
    print(df[["configuration", "n_features", "f1", "roc_auc", "fpr"]].to_string(index=False))

    # Compute contribution of each novel contribution
    full_f1  = df[df["configuration"].str.startswith("6.")]["f1"].values[0]
    base_f1  = df[df["configuration"].str.startswith("1.")]["f1"].values[0]
    peer_f1  = df[df["configuration"].str.startswith("2.")]["f1"].values[0]
    no_peer  = df[df["configuration"].str.startswith("8.")]["f1"].values[0]
    no_genai = df[df["configuration"].str.startswith("9.")]["f1"].values[0]

    print(f"\n--- Novel Contribution Impact ---")
    print(f"NC1 (Peer features) added  : +{(peer_f1 - base_f1)*100:.2f}% F1")
    print(f"NC1 (Peer features) removed: -{(full_f1 - no_peer)*100:.2f}% F1 drop")
    print(f"NC4 (GenAI features) removed: -{(full_f1 - no_genai)*100:.2f}% F1 drop")