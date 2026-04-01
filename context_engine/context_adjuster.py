import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.constants import (
    PROCESSED_DIR,
    METADATA_DIR,
    CLUSTERING_FEATURES,
    ANOMALY_THRESHOLD_K,
)
from utils.logger import get_logger

log = get_logger("context_adjuster")


# ── Risk multipliers for context signals ─────────────────
MULTIPLIERS = {
    "after_hours"      : 1.4,
    "drift_active"     : 1.3,
    "high_risk_cluster": 1.5,  # clusters with >60% attack rate
    "peer_outlier"     : 1.2,
    "threshold_breach" : 1.3,
}

HIGH_RISK_CLUSTER_THRESHOLD = 0.60   # attack rate above this = high risk


def load_baselines() -> dict:
    """Load saved baselines and thresholds from baseline_model."""
    path = METADATA_DIR / "cluster_baselines.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Baselines not found at {path}. "
            "Run context_engine/baseline_model.py first."
        )
    data = joblib.load(path)
    log.info("Baselines loaded successfully")
    return data


def load_cluster_profiles() -> pd.DataFrame:
    """Load cluster profiles to identify high-risk clusters."""
    path = METADATA_DIR / "cluster_profiles.csv"
    return pd.read_csv(path)


def get_cluster_risk_level(
    cluster_id: int,
    profiles_df: pd.DataFrame,
) -> str:
    """
    Returns 'high', 'medium', or 'low' based on cluster attack rate.
    """
    row = profiles_df[profiles_df["cluster"] == cluster_id]
    if row.empty:
        return "unknown"
    attack_rate = row["attack_rate"].values[0] / 100.0
    if attack_rate >= HIGH_RISK_CLUSTER_THRESHOLD:
        return "high"
    elif attack_rate >= 0.30:
        return "medium"
    return "low"


def compute_threshold_breach_score(
    row: pd.Series,
    thresholds: dict,
    cluster_id: int,
) -> tuple:
    """
    Check how many features breach their cluster thresholds.

    Returns:
        (breach_score, breached_features_list)
        breach_score = breaches / total_features checked
    """
    cluster_thresh = thresholds.get(cluster_id, {})
    if not cluster_thresh:
        return 0.0, []

    breaches = []
    checked  = 0

    for feat, bounds in cluster_thresh.items():
        if feat not in row.index:
            continue
        val = row[feat]
        checked += 1
        if val > bounds["upper"] or val < bounds["lower"]:
            direction = "high" if val > bounds["upper"] else "low"
            deviation = abs(val - bounds["mean"]) / (bounds["std"] + 1e-9)
            breaches.append({
                "feature"  : feat,
                "value"    : round(float(val), 4),
                "mean"     : bounds["mean"],
                "upper"    : bounds["upper"],
                "lower"    : bounds["lower"],
                "direction": direction,
                "deviation": round(deviation, 2),
            })

    breach_score = len(breaches) / checked if checked > 0 else 0.0
    return breach_score, breaches


def build_alert_reason(
    breaches       : list,
    cluster_risk   : str,
    is_after_hours : bool,
    drift_active   : bool,
    peer_score     : float,
) -> str:
    """
    Build a human-readable alert reason string.
    This is the proto-NLG layer that feeds the explainability dashboard.

    Example output:
    'Alert: src_bytes is 7.3x above cluster mean [high cluster risk]
     [after-hours access] [concept drift active]'
    """
    parts = []

    # Top breaching features (max 3, sorted by deviation)
    top = sorted(breaches, key=lambda x: -x["deviation"])[:3]
    for b in top:
        parts.append(
            f"{b['feature']} is {b['deviation']:.1f}x above "
            f"cluster mean ({b['direction']})"
        )

    if cluster_risk == "high":
        parts.append("high-risk cluster")
    elif cluster_risk == "medium":
        parts.append("medium-risk cluster")

    if is_after_hours:
        parts.append("after-hours activity")

    if drift_active:
        parts.append("concept drift detected in window")

    if peer_score > 0.4:
        parts.append(f"peer outlier score={peer_score:.2f}")

    if not parts:
        return "No significant anomaly signals"

    return " | ".join(parts)


def adjust_risk_score(
    base_score     : float,
    cluster_risk   : str,
    breach_score   : float,
    is_after_hours : bool,
    drift_active   : bool,
    peer_score     : float,
) -> float:
    """
    Apply context multipliers to a base anomaly score.

    Args:
        base_score     : raw model score [0, 1]
        cluster_risk   : 'high' / 'medium' / 'low'
        breach_score   : fraction of features breaching thresholds
        is_after_hours : bool
        drift_active   : bool
        peer_score     : peer_outlier_score value

    Returns:
        context-adjusted score, clipped to [0, 1]
    """
    score = base_score

    if cluster_risk == "high":
        score *= MULTIPLIERS["high_risk_cluster"]
    elif cluster_risk == "medium":
        score *= 1.2

    if breach_score > 0.3:
        score *= MULTIPLIERS["threshold_breach"]

    if is_after_hours:
        score *= MULTIPLIERS["after_hours"]

    if drift_active:
        score *= MULTIPLIERS["drift_active"]

    if peer_score > 0.4:
        score *= MULTIPLIERS["peer_outlier"]

    return float(np.clip(score, 0.0, 1.0))


def run_context_adjustment(
    df            : pd.DataFrame,
    baselines_data: dict = None,
) -> pd.DataFrame:
    """
    Master function. Applies context adjustment to every row.

    Adds columns:
      - cluster_risk_level   : 'high' / 'medium' / 'low'
      - breach_score         : fraction of threshold breaches
      - context_risk_score   : adjusted risk score [0, 1]
      - alert_reason         : human-readable explanation string

    Args:
        df             : DataFrame with cluster + feature columns
        baselines_data : output of load_baselines() — loads if None

    Returns:
        df with context columns added
    """
    if baselines_data is None:
        baselines_data = load_baselines()

    thresholds   = baselines_data["thresholds"]
    profiles_df  = load_cluster_profiles()

    log.info(f"Running context adjustment on {len(df):,} rows...")

    cluster_risk_levels = []
    breach_scores       = []
    context_scores      = []
    alert_reasons       = []

    # Use peer_outlier_score and any_drift if present
    has_drift      = "any_drift" in df.columns
    has_peer_score = "peer_outlier_score" in df.columns

    for idx, row in df.iterrows():
        cluster_id   = int(row.get("cluster", 0))
        cluster_risk = get_cluster_risk_level(cluster_id, profiles_df)

        breach_score, breaches = compute_threshold_breach_score(
            row, thresholds, cluster_id
        )

        is_after_hours = bool(row.get("is_after_hours", 0))
        drift_active   = bool(row["any_drift"]) if has_drift else False
        peer_score     = float(row["peer_outlier_score"]) \
                         if has_peer_score else 0.0

        # Base score = breach_score as proxy
        # (will be replaced by model output in Phase 3)
        base_score = breach_score

        context_score = adjust_risk_score(
            base_score, cluster_risk, breach_score,
            is_after_hours, drift_active, peer_score
        )

        reason = build_alert_reason(
            breaches, cluster_risk, is_after_hours,
            drift_active, peer_score
        )

        cluster_risk_levels.append(cluster_risk)
        breach_scores.append(round(breach_score, 4))
        context_scores.append(round(context_score, 4))
        alert_reasons.append(reason)

    df["cluster_risk_level"] = cluster_risk_levels
    df["breach_score"]       = breach_scores
    df["context_risk_score"] = context_scores
    df["alert_reason"]       = alert_reasons

    # Summary
    high_risk = (df["context_risk_score"] > 0.5).sum()
    log.info(f"Context adjustment complete - "
             f"{high_risk:,} rows scored above 0.5 "
             f"({high_risk/len(df)*100:.1f}%)")

    return df


if __name__ == "__main__":
    # Load features
    path = PROCESSED_DIR / "nslkdd_train_features.csv"
    df   = pd.read_csv(path)

    # Assign clusters
    scaler = joblib.load(METADATA_DIR / "cluster_scaler.pkl")
    km     = joblib.load(METADATA_DIR / "kmeans_model.pkl")
    X      = df[[f for f in CLUSTERING_FEATURES if f in df.columns]]
    df["cluster"] = km.predict(scaler.transform(X))

    # Run drift detection first (adds any_drift column)
    from context_engine.drift_detection import run_drift_detection
    df, _ = run_drift_detection(df)

    # Run context adjustment
    df = run_context_adjustment(df)

    # Show results
    print("\nContext Risk Score Distribution:")
    print(df["context_risk_score"].describe().to_string())

    print("\nSample high-risk alerts:")
    high = df[df["context_risk_score"] > 0.5][
        ["cluster", "cluster_risk_level",
         "breach_score", "context_risk_score",
         "alert_reason", "label_binary"]
    ].head(8)
    print(high.to_string())

    # Save
    out = PROCESSED_DIR / "nslkdd_train_context.csv"
    df.to_csv(out, index=False)
    log.info(f"Saved -> {out}")