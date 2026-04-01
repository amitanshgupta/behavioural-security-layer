import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import joblib
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.constants import (
    PROCESSED_DIR,
    METADATA_DIR,
    CLUSTERING_FEATURES,
    ROLLING_WINDOWS,
    ANOMALY_THRESHOLD_K,
)
from utils.logger import get_logger

log = get_logger("baseline_model")

BASELINE_PATH = METADATA_DIR / "cluster_baselines.pkl"


def compute_cluster_baselines(
    df: pd.DataFrame,
    features: list[str],
) -> dict:
    """
    For each cluster, compute mean and std of each feature.
    This is the static baseline — the foundation before rolling windows.

    Returns dict:
      {cluster_id: {feature: {"mean": float, "std": float}}}
    """
    baselines = {}

    for cluster_id in sorted(df["cluster"].unique()):
        grp = df[df["cluster"] == cluster_id]
        baselines[cluster_id] = {}

        for feat in features:
            if feat not in grp.columns:
                continue
            mean = grp[feat].mean()
            std  = grp[feat].std()
            baselines[cluster_id][feat] = {
                "mean"  : round(mean, 6),
                "std"   : round(std, 6),
                "count" : len(grp),
            }

        log.info(
            f"Cluster {cluster_id:>2} baseline computed "
            f"({len(grp):>7,} samples, "
            f"{len(baselines[cluster_id])} features)"
        )

    return baselines


def compute_rolling_baselines(
    df: pd.DataFrame,
    features: list[str],
    windows: list[int] = ROLLING_WINDOWS,
) -> dict:
    """
    Simulate rolling window baselines using row-order as time proxy.
    (NSL-KDD has no timestamps — rows are ordered temporally by design)

    For each window size, for each cluster, compute:
      - rolling mean of each feature over last N samples
      - rolling std of each feature over last N samples

    Returns dict:
      {window: {cluster_id: {feature: {"mean": float, "std": float}}}}

    NOTE: With CERT data, replace row-order with actual timestamps.
    """
    rolling_baselines = {}

    for window in windows:
        rolling_baselines[window] = {}
        log.info(f"Computing rolling baseline for window={window}...")

        for cluster_id in sorted(df["cluster"].unique()):
            grp = df[df["cluster"] == cluster_id].copy()

            if len(grp) < window:
                log.warning(
                    f"Cluster {cluster_id} has only {len(grp)} samples "
                    f"— less than window={window}, using all available"
                )
                tail = grp
            else:
                tail = grp.tail(window)

            rolling_baselines[window][cluster_id] = {}
            for feat in features:
                if feat not in tail.columns:
                    continue
                rolling_baselines[window][cluster_id][feat] = {
                    "mean" : round(tail[feat].mean(), 6),
                    "std"  : round(tail[feat].std(), 6),
                    "count": len(tail),
                }

    log.info(f"Rolling baselines computed for windows: {windows}")
    return rolling_baselines


def compute_dynamic_thresholds(
    baselines: dict,
    k: float = ANOMALY_THRESHOLD_K,
) -> dict:
    """
    For each cluster + feature, compute anomaly thresholds:
      upper = mean + k * std
      lower = mean - k * std  (floored at 0 for non-negative features)

    Args:
        baselines : output of compute_cluster_baselines()
        k         : multiplier (default 3.0 from constants)

    Returns dict:
      {cluster_id: {feature: {"upper": float, "lower": float}}}
    """
    thresholds = {}

    for cluster_id, feats in baselines.items():
        thresholds[cluster_id] = {}
        for feat, stats in feats.items():
            mean = stats["mean"]
            std  = stats["std"]
            thresholds[cluster_id][feat] = {
                "upper": round(mean + k * std, 6),
                "lower": round(max(0.0, mean - k * std), 6),
                "mean" : mean,
                "std"  : std,
            }

    log.info(
        f"Dynamic thresholds computed (k={k}) "
        f"for {len(thresholds)} clusters"
    )
    return thresholds


def flag_threshold_violations(
    df: pd.DataFrame,
    thresholds: dict,
    features: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    For each row, check if any feature exceeds its cluster threshold.
    Adds:
      - {feat}_violation : binary flag per feature
      - violation_count  : total violations per row
      - violation_score  : violation_count / n_features checked (0-1)
    """
    features = features or CLUSTERING_FEATURES
    features = [f for f in features if f in df.columns]

    violation_cols = []

    for feat in features:
        col_name = f"{feat}_violation"
        violations = []

        for _, row in df.iterrows():
            cluster_id = row.get("cluster", -1)
            if cluster_id not in thresholds:
                violations.append(0)
                continue

            thresh = thresholds[cluster_id].get(feat, {})
            if not thresh:
                violations.append(0)
                continue

            val = row[feat]
            if val > thresh["upper"] or val < thresh["lower"]:
                violations.append(1)
            else:
                violations.append(0)

        df[col_name] = violations
        violation_cols.append(col_name)

    df["violation_count"] = df[violation_cols].sum(axis=1)
    df["violation_score"] = df["violation_count"] / len(features)

    flagged_pct = (df["violation_count"] > 0).mean() * 100
    log.info(
        f"Threshold violations computed — "
        f"{flagged_pct:.1f}% of rows have at least 1 violation"
    )
    return df


def run_baseline_model(df: pd.DataFrame) -> tuple[dict, dict, dict]:
    """
    Master function. Computes and saves all baselines and thresholds.

    Returns:
        (static_baselines, rolling_baselines, thresholds)
    """
    log.info(f"Running baseline model on {len(df):,} rows...")

    features = [f for f in CLUSTERING_FEATURES if f in df.columns]

    static   = compute_cluster_baselines(df, features)
    rolling  = compute_rolling_baselines(df, features)
    thresh   = compute_dynamic_thresholds(static)

    # Save all
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"static": static, "rolling": rolling, "thresholds": thresh},
        BASELINE_PATH
    )
    log.info(f"Baselines saved -> {BASELINE_PATH}")

    return static, rolling, thresh


if __name__ == "__main__":
    # Load clustered train data
    path = PROCESSED_DIR / "nslkdd_train_features.csv"
    df   = pd.read_csv(path)

    # Load cluster assignments
    import joblib as jl
    from context_engine.clustering import assign_clusters_to_test
    scaler = jl.load(METADATA_DIR / "cluster_scaler.pkl")
    km     = jl.load(METADATA_DIR / "kmeans_model.pkl")

    from sklearn.preprocessing import StandardScaler
    from utils.constants import CLUSTERING_FEATURES
    X        = df[[f for f in CLUSTERING_FEATURES if f in df.columns]]
    X_scaled = scaler.transform(X)
    df["cluster"] = km.predict(X_scaled)

    static, rolling, thresholds = run_baseline_model(df)

    # Show thresholds for cluster 0 on a few features
    print("\nSample thresholds for Cluster 0:")
    sample_feats = ["src_bytes", "duration", "count",
                    "error_rate_combined", "peer_outlier_score"]
    for feat in sample_feats:
        t = thresholds[0].get(feat, {})
        if t:
            print(f"  {feat:<30} "
                  f"lower={t['lower']:.4f}  "
                  f"mean={t['mean']:.4f}  "
                  f"upper={t['upper']:.4f}")

    # Show rolling window sizes per cluster
    print("\nRolling window sample counts (window=30, cluster 0):")
    for feat in sample_feats:
        r = rolling[30][0].get(feat, {})
        if r:
            print(f"  {feat:<30} count={r['count']}  "
                  f"mean={r['mean']:.4f}")